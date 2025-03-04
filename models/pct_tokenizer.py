# --------------------------------------------------------
# Pose Compositional Tokens
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmpose.models.builder import build_loss
from mmpose.models.builder import HEADS
from timm.models.layers import trunc_normal_

from .modules import MixerLayer
    

@HEADS.register_module()
class PCT_Tokenizer(nn.Module):
    """ Tokenizer of Pose Compositional Tokens.
        paper ref: Zigang Geng et al. "Human Pose as
            Compositional Tokens"

    Args:
        stage_pct (str): Training stage (Tokenizer or Classifier).
        tokenizer (list): Config about the tokenizer.
        num_joints (int): Number of annotated joints in the dataset.
        guide_ratio (float): The ratio of image guidance.
        guide_channels (int): Feature Dim of the image guidance.
    """

    def __init__(self,
                 stage_pct,
                 tokenizer=None,
                 num_joints=17,
                 guide_ratio=0,
                 guide_channels=0):
        super().__init__()

        self.stage_pct = stage_pct
        self.guide_ratio = guide_ratio
        self.num_joints = num_joints

        self.drop_rate = tokenizer['encoder']['drop_rate']     
        self.enc_num_blocks = tokenizer['encoder']['num_blocks']
        self.enc_hidden_dim = tokenizer['encoder']['hidden_dim']
        self.enc_token_inter_dim = tokenizer['encoder']['token_inter_dim']
        self.enc_hidden_inter_dim = tokenizer['encoder']['hidden_inter_dim']
        self.enc_dropout = tokenizer['encoder']['dropout']

        self.dec_num_blocks = tokenizer['decoder']['num_blocks']
        self.dec_hidden_dim = tokenizer['decoder']['hidden_dim']
        self.dec_token_inter_dim = tokenizer['decoder']['token_inter_dim']
        self.dec_hidden_inter_dim = tokenizer['decoder']['hidden_inter_dim']
        self.dec_dropout = tokenizer['decoder']['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']
        self.token_dim = tokenizer['codebook']['token_dim']
        self.decay = tokenizer['codebook']['ema_decay']

        self.invisible_token = nn.Parameter(
            torch.zeros(1, 1, self.enc_hidden_dim))
        trunc_normal_(self.invisible_token, mean=0., std=0.02, a=-0.02, b=0.02)

        if self.guide_ratio > 0:
            self.start_img_embed = nn.Linear(
                guide_channels, int(self.enc_hidden_dim*self.guide_ratio))
        self.start_embed = nn.Linear(
            2, int(self.enc_hidden_dim*(1-self.guide_ratio)))
        
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim, 
                self.num_joints, self.enc_token_inter_dim,
                self.enc_dropout) for _ in range(self.enc_num_blocks)])
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)
        
        self.token_mlp = nn.Linear(
            self.num_joints, self.token_num)
        self.feature_embed = nn.Linear(
            self.enc_hidden_dim, self.token_dim)

        self.register_buffer('codebook', 
            torch.empty(self.token_class_num, self.token_dim))
        self.codebook.data.normal_()
        self.register_buffer('ema_cluster_size', 
            torch.zeros(self.token_class_num))
        self.register_buffer('ema_w', 
            torch.empty(self.token_class_num, self.token_dim))
        self.ema_w.data.normal_()        
        
        self.decoder_token_mlp = nn.Linear(
            self.token_num, self.num_joints)
        self.decoder_start = nn.Linear(
            self.token_dim, self.dec_hidden_dim)

        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 2)

        self.loss = build_loss(tokenizer['loss_keypoint'])

    def forward(self, joints, joints_feature, cls_logits, train=True):
        """Forward function. """

        if train or self.stage_pct == "tokenizer":
            # Encoder of Tokenizer, Get the PCT groundtruth class labels.
            joints_coord, joints_visible, bs \
                = joints[:,:,:-1], joints[:,:,-1].bool(), joints.shape[0]

            encode_feat = self.start_embed(joints_coord)
            if self.guide_ratio > 0:
                encode_img_feat = self.start_img_embed(joints_feature)
                encode_feat = torch.cat((encode_feat, encode_img_feat), dim=2)

            if train and self.stage_pct == "tokenizer":
                rand_mask_ind = torch.rand(
                    joints_visible.shape, device=joints.device) > self.drop_rate
                joints_visible = torch.logical_and(rand_mask_ind, joints_visible) 

            mask_tokens = self.invisible_token.expand(bs, joints.shape[1], -1)
            w = joints_visible.unsqueeze(-1).type_as(mask_tokens)
            encode_feat = encode_feat * w + mask_tokens * (1 - w)
                    
            for num_layer in self.encoder:
                encode_feat = num_layer(encode_feat)
            encode_feat = self.encoder_layer_norm(encode_feat)
            
            encode_feat = encode_feat.transpose(2, 1)
            encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
            encode_feat = self.feature_embed(encode_feat).flatten(0,1)
            
            distances = torch.sum(encode_feat**2, dim=1, keepdim=True) \
                + torch.sum(self.codebook**2, dim=1) \
                - 2 * torch.matmul(encode_feat, self.codebook.t())
                
            encoding_indices = torch.argmin(distances, dim=1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self.token_class_num, device=joints.device)
            encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        else:
            bs = cls_logits.shape[0] // self.token_num
            encoding_indices = None
            distances = None
        
        if self.stage_pct == "classifier":
            part_token_feat = torch.matmul(cls_logits, self.codebook)
        else:
            part_token_feat = torch.matmul(encodings, self.codebook)

        if train and self.stage_pct == "tokenizer":
            # Updating Codebook using EMA
            dw = torch.matmul(encodings.t(), encode_feat.detach())
            # sync
            n_encodings, n_dw = encodings.numel(), dw.numel()
            encodings_shape, dw_shape = encodings.shape, dw.shape
            combined = torch.cat((encodings.flatten(), dw.flatten()))
            if dist.is_initialized():
                dist.all_reduce(combined) # math sum
            sync_encodings, sync_dw = torch.split(combined, [n_encodings, n_dw])
            sync_encodings, sync_dw = \
                sync_encodings.view(encodings_shape), sync_dw.view(dw_shape)

            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(sync_encodings, 0)
            
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.token_class_num * 1e-5) * n)
            
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * sync_dw
            self.codebook = self.ema_w / self.ema_cluster_size.unsqueeze(1)
            e_latent_loss = F.mse_loss(part_token_feat.detach(), encode_feat)
            part_token_feat = encode_feat + (part_token_feat - encode_feat).detach()
        else:
            e_latent_loss = None
        
        # Decoder of Tokenizer, Recover the joints.
        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)
        
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_joints = self.recover_embed(decode_feat)

        return recoverd_joints, encoding_indices, e_latent_loss, distances
    
    def decode_codewords(self,encoding_indices=None,cls_logits=None):
        bs = None
        if cls_logits is None:
            assert encoding_indices is not None
            encoding_indices = encoding_indices.view(-1)
            cls_logits = torch.zeros(
                encoding_indices.shape[0], self.token_class_num, device=encoding_indices.device)
            cls_logits.scatter_(1, encoding_indices.unsqueeze(1), 1)
        bs = cls_logits.shape[0] // self.token_num
        
        part_token_feat = torch.matmul(cls_logits, self.codebook)
        
        # Decoder of Tokenizer, Recover the joints.
        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)
        
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_joints = self.recover_embed(decode_feat)
        return recoverd_joints

    def get_loss(self, output_joints, joints, e_latent_loss):
        """Calculate loss for training tokenizer.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output_joints (torch.Tensor[NxKx3]): Recovered joints.
            joints(torch.Tensor[NxKx3]): Target joints.
            e_latent_loss(torch.Tensor[1]): Loss for training codebook.
        """

        losses = dict()

        kpt_loss, e_latent_loss = self.loss(output_joints, joints, e_latent_loss)

        losses['joint_loss'] = kpt_loss
        losses['e_latent_loss'] = e_latent_loss

        return losses

    def init_weights(self, pretrained=""):
        """Initialize model weights."""

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            assert (self.stage_pct == "classifier"), \
                "Training tokenizer does not need to load model"
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)

            need_init_state_dict = {}

            for name, m in pretrained_state_dict['state_dict'].items():
                if 'keypoint_head.tokenizer.' in name:
                    name = name.replace('keypoint_head.tokenizer.', '')
                if name in parameters_names or name in buffers_names:
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=True)
        else:
            if self.stage_pct == "classifier":
                print('If you are training a classifier, '\
                    'must check that the well-trained tokenizer '\
                    'is located in the correct path.')

# added by kristin
from vqtorch.nn import VectorQuant

class PCT_Tokenizer_STE(nn.Module):
    """ Tokenizer of Pose Compositional Tokens.
        paper ref: Zigang Geng et al. "Human Pose as
            Compositional Tokens"

    Args:
        stage_pct (str): Training stage (Tokenizer or Classifier).
        tokenizer (list): Config about the tokenizer.
        num_joints (int): Number of annotated joints in the dataset.
        guide_ratio (float): The ratio of image guidance.
        guide_channels (int): Feature Dim of the image guidance.
    """

    def __init__(self,
                 stage_pct,
                 tokenizer=None,
                 num_joints=17,
                 guide_ratio=0,
                 guide_channels=0):
        super().__init__()

        self.stage_pct = stage_pct
        self.guide_ratio = guide_ratio
        self.num_joints = num_joints

        self.drop_rate = tokenizer['encoder']['drop_rate']     
        self.enc_num_blocks = tokenizer['encoder']['num_blocks']
        self.enc_hidden_dim = tokenizer['encoder']['hidden_dim']
        self.enc_token_inter_dim = tokenizer['encoder']['token_inter_dim']
        self.enc_hidden_inter_dim = tokenizer['encoder']['hidden_inter_dim']
        self.enc_dropout = tokenizer['encoder']['dropout']

        self.dec_num_blocks = tokenizer['decoder']['num_blocks']
        self.dec_hidden_dim = tokenizer['decoder']['hidden_dim']
        self.dec_token_inter_dim = tokenizer['decoder']['token_inter_dim']
        self.dec_hidden_inter_dim = tokenizer['decoder']['hidden_inter_dim']
        self.dec_dropout = tokenizer['decoder']['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']
        self.token_dim = tokenizer['codebook']['token_dim']
        self.decay = tokenizer['codebook']['ema_decay']
        
        if 'vqste' in tokenizer['codebook']:
            vq_kwargs = tokenizer['codebook']['vqste']
        else:
            vq_kwargs = {}
        if 'share_codebook' in tokenizer['codebook'] and tokenizer['codebook']['share_codebook']:
            self.share_codebook = True
        else:
            self.share_codebook = False

        self.invisible_token = nn.Parameter(
            torch.zeros(1, 1, self.enc_hidden_dim))
        trunc_normal_(self.invisible_token, mean=0., std=0.02, a=-0.02, b=0.02)

        if self.guide_ratio > 0:
            self.start_img_embed = nn.Linear(
                guide_channels, int(self.enc_hidden_dim*self.guide_ratio))
        self.start_embed = nn.Linear(
            2, int(self.enc_hidden_dim*(1-self.guide_ratio)))
        
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim, 
                self.num_joints, self.enc_token_inter_dim,
                self.enc_dropout) for _ in range(self.enc_num_blocks)])
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)
        
        self.token_mlp = nn.Linear(
            self.num_joints, self.token_num)
        self.feature_embed = nn.Linear(
            self.enc_hidden_dim, self.token_dim)

        if self.share_codebook:
            self.vqste = VectorQuant(feature_size=self.token_dim,
                                    num_codes=self.token_class_num,
                                    **vq_kwargs)
        else:
            self.vqste = nn.ModuleList()
            for _ in range(self.token_num):
                self.vqste.append(VectorQuant(feature_size=self.token_dim,
                                    num_codes=self.token_class_num,
                                    **vq_kwargs))
                
        # self.register_buffer('codebook', 
        #     torch.empty(self.token_class_num, self.token_dim))
        # self.codebook.data.normal_()
        # self.register_buffer('ema_cluster_size', 
        #     torch.zeros(self.token_class_num))
        # self.register_buffer('ema_w', 
        #     torch.empty(self.token_class_num, self.token_dim))
        # self.ema_w.data.normal_()        
        
        self.decoder_token_mlp = nn.Linear(
            self.token_num, self.num_joints)
        self.decoder_start = nn.Linear(
            self.token_dim, self.dec_hidden_dim)

        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 2)

        self.loss = build_loss(tokenizer['loss_keypoint'])

    @property
    def codebook(self):
        if self.share_codebook:
            return self.vqste.get_codebook()
        else:
            return [vqste.get_codebook() for vqste in self.vqste]

    def forward(self, joints, joints_feature, cls_logits, train=True):
        """Forward function. """

        if self.share_codebook:
            if train:
                self.vqste.train()
            else:
                self.vqste.eval()
        else:
            for vqste in self.vqste:
                if train:
                    vqste.train()
                else:
                    vqste.eval()

        if train or self.stage_pct == "tokenizer":
            # Encoder of Tokenizer, Get the PCT groundtruth class labels.
            joints_coord, joints_visible, bs \
                = joints[:,:,:-1], joints[:,:,-1].bool(), joints.shape[0]

            encode_feat = self.start_embed(joints_coord)
            if self.guide_ratio > 0:
                encode_img_feat = self.start_img_embed(joints_feature)
                encode_feat = torch.cat((encode_feat, encode_img_feat), dim=2)

            if train and self.stage_pct == "tokenizer":
                rand_mask_ind = torch.rand(
                    joints_visible.shape, device=joints.device) > self.drop_rate
                joints_visible = torch.logical_and(rand_mask_ind, joints_visible) 

            mask_tokens = self.invisible_token.expand(bs, joints.shape[1], -1)
            w = joints_visible.unsqueeze(-1).type_as(mask_tokens)
            encode_feat = encode_feat * w + mask_tokens * (1 - w)
                    
            for num_layer in self.encoder:
                encode_feat = num_layer(encode_feat) 
            encode_feat = self.encoder_layer_norm(encode_feat) # bs x njoints x enc_hidden_dim
            
            encode_feat = encode_feat.transpose(2, 1) # bs x enc_hidden_dim x njoints
            encode_feat = self.token_mlp(encode_feat).transpose(2, 1) # bs x token_num x enc_hidden_dim
            encode_feat = self.feature_embed(encode_feat) # bs x token_num x enc_hidden_dim
            if self.share_codebook:
                encode_feat = encode_feat.flatten(0,1) # bs*token_num x token_dim
                part_token_feat,vq_dict = self.vqste(encode_feat) 
                # part_token_feat: bs*token_num x token_dim
                # vq_dict['z']: bs*token_num x 1 x token_dim
                # vq_dict['z_q']: bs*token_num x 1 x token_dim
                # vq_dict['d']: bs*token_num x 1
                # vq_dict['q']: bs*token_num x 1
                # vq_dict['loss']: torch.Size([])
                # vq_dict['perplexity']: None
            else:
                part_token_feats = []
                for tokeni in range(self.token_num):
                    part_token_feat_curr,vq_dict_curr = self.vqste[tokeni](encode_feat[:,tokeni])
                    part_token_feats.append(part_token_feat_curr)
                    if tokeni == 0:
                        vq_dicts = {k: [v,] for k,v in vq_dict_curr.items()}
                    else:
                        for k,v in vq_dict_curr.items():
                            vq_dicts[k].append(v)
                part_token_feat = torch.cat([p.unsqueeze(1) for p in part_token_feats], dim=1).flatten(0,1)
                vq_dict = {'d': torch.cat([vv.unsqueeze(1) for vv in vq_dicts['d']],dim=1).flatten(0,1),
                           'q': torch.cat([vv.unsqueeze(1) for vv in vq_dicts['q']],dim=1).flatten(0,1),
                           'loss': sum([vv for vv in vq_dicts['loss']])}
                
            distances = vq_dict['d'][...,0]
            encoding_indices = vq_dict['q'][...,0]
            e_latent_loss = vq_dict['loss']
            encodings = torch.zeros(
                encoding_indices.shape[0], self.token_class_num, device=joints.device)
            encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        else:
            bs = cls_logits.shape[0] // self.token_num
            encoding_indices = None
            distances = None
            e_latent_loss = None
        
        if self.stage_pct == "classifier":
            if self.share_codebook:
                part_token_feat = torch.matmul(cls_logits, self.codebook)
            else:
                cls_logits = cls_logits.view(bs, self.token_num, -1)
                part_token_feats = []
                for tokeni in range(self.token_num):
                    part_token_feat_curr = torch.matmul(cls_logits[:,tokeni], self.codebook[tokeni])
                    part_token_feats.append(part_token_feat_curr)
                part_token_feat = torch.cat([p.unsqueeze(1) for p in part_token_feats], dim=1).flatten(0,1)
        
        # Decoder of Tokenizer, Recover the joints.
        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)
        
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_joints = self.recover_embed(decode_feat)

        return recoverd_joints, encoding_indices, e_latent_loss, distances
    
    def decode_codewords(self,encoding_indices=None,cls_logits=None):
        bs = None
        if cls_logits is None:
            assert encoding_indices is not None
            encoding_indices = encoding_indices.view(-1)
            cls_logits = torch.zeros(
                encoding_indices.shape[0], self.token_class_num, device=encoding_indices.device)
            cls_logits.scatter_(1, encoding_indices.unsqueeze(1), 1)
        bs = cls_logits.shape[0] // self.token_num
        
        if self.share_codebook:
            part_token_feat = torch.matmul(cls_logits, self.codebook)
        else:
            cls_logits = cls_logits.view(bs, self.token_num, -1)
            part_token_feats = []
            for tokeni in range(self.token_num):
                part_token_feat_curr = torch.matmul(cls_logits[:,tokeni], self.codebook[tokeni])
                part_token_feats.append(part_token_feat_curr)
            part_token_feat = torch.cat([p.unsqueeze(1) for p in part_token_feats], dim=1).flatten(0,1)
                    
        # Decoder of Tokenizer, Recover the joints.
        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)
        
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_joints = self.recover_embed(decode_feat)
        return recoverd_joints

    def get_loss(self, output_joints, joints, e_latent_loss):
        """Calculate loss for training tokenizer.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output_joints (torch.Tensor[NxKx3]): Recovered joints.
            joints(torch.Tensor[NxKx3]): Target joints.
            e_latent_loss(torch.Tensor[1]): Loss for training codebook.
        """

        losses = dict()

        kpt_loss, e_latent_loss = self.loss(output_joints, joints, e_latent_loss)

        losses['joint_loss'] = kpt_loss
        losses['e_latent_loss'] = e_latent_loss

        return losses

    def init_weights(self, pretrained=""):
        """Initialize model weights."""

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            assert (self.stage_pct == "classifier"), \
                "Training tokenizer does not need to load model"
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)

            need_init_state_dict = {}

            for name, m in pretrained_state_dict['state_dict'].items():
                if 'keypoint_head.tokenizer.' in name:
                    name = name.replace('keypoint_head.tokenizer.', '')
                if name in parameters_names or name in buffers_names:
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=True)
        else:
            if self.stage_pct == "classifier":
                print('If you are training a classifier, '\
                    'must check that the well-trained tokenizer '\
                    'is located in the correct path.')
