# --------------------------------------------------------
# Pose Compositional Tokens
# Based on MMPose (https://github.com/open-mmlab/mmpose)
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import time
import torch
import numpy as np
import warnings

import mmcv
from mmcv.runner import auto_fp16
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose
from mmpose.core.post_processing import transform_preds


def keypoint_info_to_flip_idx(keypoint_info):
    
    keypoint_name_to_idx = {}
    for k,v in keypoint_info.items():
        keypoint_name_to_idx[v['name']] = k
    flipidx = np.zeros(len(keypoint_info),dtype=int)
    for idx,v in keypoint_info.items():
        flipidx[idx] = keypoint_name_to_idx[v['swap']]
        
    return flipidx.tolist()


@POSENETS.register_module()
class PCT(BasePose):
    """ Detector of Pose Compositional Tokens.
        paper ref: Zigang Geng et al. "Human Pose as
            Compositional Tokens"

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.stage_pct = keypoint_head['stage_pct']
        assert self.stage_pct in ["tokenizer", "classifier"]
        self.image_guide = keypoint_head['tokenizer']['guide_ratio'] > 0

        if self.stage_pct == "tokenizer":
            # For training tokenizer
            keypoint_head['loss_keypoint'] \
                = keypoint_head['tokenizer']['loss_keypoint']

        if self.stage_pct == "classifier":
            # For training classifier
            # backbone is only needed for training classifier
            self.backbone = builder.build_backbone(backbone)

        if self.image_guide:
            # extra_backbone is optional feature to guide the training tokenizer
            # It brings a slight impact on performance
            self.extra_backbone = builder.build_backbone(backbone)
            
        self.keypoint_head = builder.build_head(keypoint_head)

        self.init_weights(pretrained, keypoint_head['tokenizer']['ckpt'])

        self.flip_test = test_cfg.get('flip_test', True)
        self.dataset_name = test_cfg.get('dataset_name', 'COCO')
        
        keypoint_info = test_cfg.get('keypoint_info', None)
        if keypoint_info is None:
            self.flipidx = None
            if self.flip_test:
                warnings.warn(f"keypoint_head.keypoint_info is not set, setting flip index based on dataset_name: {self.dataset_name}")
        else:
            self.flipidx = keypoint_info_to_flip_idx(keypoint_info)


    def init_weights(self, pretrained, tokenizer):
        """Weight initialization for model."""
        if self.stage_pct == "classifier":
            self.backbone.init_weights(pretrained)
        if self.image_guide:
            self.extra_backbone.init_weights(pretrained)
        self.keypoint_head.init_weights()
        self.keypoint_head.tokenizer.init_weights(tokenizer)

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                joints_3d=None,
                joints_3d_visible=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img width: imgW

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            joints_3d (torch.Tensor[NxKx3]): Target joints.
            joints_3d_visible (torch.Tensor[NxKx3]): Visibility of each target joint.
                Only first NxKx1 is valid.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes, image paths.
        """
        if return_loss or self.stage_pct == "tokenizer":
            joints = joints_3d
            joints[...,-1] = joints_3d_visible[...,0]
        else:
            # Just a placeholder during inference of PCT
            joints = None

        if return_loss:
            return self.forward_train(img, joints, img_metas, **kwargs)
        return self.forward_test(
            img, joints, img_metas, **kwargs)

    def forward_train(self, img, joints, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""

        output = None if self.stage_pct == "tokenizer" else self.backbone(img)
        extra_output = self.extra_backbone(img) if self.image_guide else None

        p_logits, p_joints, g_logits, e_latent_loss = \
            self.keypoint_head(output, extra_output, joints)

        # if return loss
        losses = dict()
        if self.stage_pct == "classifier":
            keypoint_losses = self.keypoint_head.get_loss(
                p_logits, p_joints, g_logits, joints)
            losses.update(keypoint_losses)

            topk = (1,2,5)
            keypoint_accuracy = \
                self.get_class_accuracy(p_logits, g_logits, topk)
            kpt_accs = {}
            for i in range(len(topk)):
                kpt_accs['top%s-acc' % str(topk[i])] \
                    = keypoint_accuracy[i]
            losses.update(kpt_accs)
        elif self.stage_pct == "tokenizer":
            keypoint_losses = \
                self.keypoint_head.tokenizer.get_loss(
                    p_joints, joints, e_latent_loss)
            losses.update(keypoint_losses)
        
        return losses

    def get_class_accuracy(self, output, target, topk):
        
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [
            correct[:k].reshape(-1).float().sum(0) \
                * 100. / batch_size for k in topk]

    def forward_test(self, img, joints, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        if (img is not None) and (img_metas is not None):
            assert img.size(0) == len(img_metas)

        results = {}
    
        # remove dependency on img
        
        if img is None:
            assert (self.stage_pct == "tokenizer") and (self.image_guide == False), "img must not be None"
            batch_size = joints.shape[0]
        else:
            batch_size = img.shape[0]
            assert img_metas is not None, "img_metas must not be None"

        if img_metas is not None:
            if batch_size > 1:
                assert 'bbox_id' in img_metas[0]
            
        output = None if self.stage_pct == "tokenizer" \
            else self.backbone(img) 
        extra_output = self.extra_backbone(img) \
            if self.image_guide and self.stage_pct == "tokenizer" else None
        
        p_joints, encoding_scores, codebook_indices, codebook_distances = \
            self.keypoint_head(output, extra_output, joints, train=False)
        score_pose = joints[:,:,2:] if self.stage_pct == "tokenizer" else \
            encoding_scores.mean(1, keepdim=True).repeat(1,p_joints.shape[1],1)

        if img is None:
            try:
                im_width = self.keypoint_head.image_size[1]
                im_height = self.keypoint_head.image_size[0]
            except:
                warnings.warn("Guessing image size from keypoints")
                # guess an image size so that the joints are centered
                joints_center_x = torch.mean(joints[...,0][joints[:,:,2]>0])
                joints_center_y = torch.mean(joints[...,1][joints[:,:,2]>0])
                max_x = torch.max(joints[...,0])
                max_y = torch.max(joints[...,0])
                im_width = int(torch.ceil(torch.maximum(joints_center_x, max_x)*2).item()+1)
                im_height = int(torch.ceil(torch.maximum(joints_center_y, max_y)*2).item()+1)
        else:
            im_width = img.shape[-1]
            im_height = img.shape[-2]

        # this is pretty hacky...
        if self.flip_test:
            if self.flipidx is None:
                FLIP_INDEX = {'COCO': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], \
                        'CROWDPOSE': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13], \
                        'OCCLUSIONPERSON':[0, 4, 5, 6, 1, 2, 3, 7, 8, 12, 13, 14, 9, 10, 11],\
                        'MPII': [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10],
                        'fly_bubble_data_20241024_17kp': [0, 2, 1, 4, 3, 5, 6, 9, 10, 7, 8, 16, 15, 14, 13, 12, 11],
                        'fly_bubble_data_20241024': [0, 2, 1, 4, 3, 5, 6, 9, 10, 7, 8, 16, 15, 14, 13, 12, 11, 19, 20, 17, 18],
                        }
                flipidx = FLIP_INDEX[self.dataset_name]
            else:
                flipidx = self.flipidx

            # remove dependence on img
            if img is None:
                img_flipped = None
            else:
                img_flipped = img.flip(3)
    
            features_flipped = None if self.stage_pct == "tokenizer" \
                else self.backbone(img_flipped) 
            extra_output_flipped = self.extra_backbone(img_flipped) \
                if self.image_guide and self.stage_pct == "tokenizer" else None

            if joints is not None:
                joints_flipped = joints.clone()
                joints_flipped = joints_flipped[:,flipidx,:]
                joints_flipped[:,:,0] = im_width - 1 - joints_flipped[:,:,0]
            else:
                joints_flipped = None
                p_joints[0,:,0].cpu().numpy()
            p_joints_f, encoding_scores_f, codebook_indices_f, codebook_distances_f = \
                self.keypoint_head(features_flipped, \
                    extra_output_flipped, joints_flipped, train=False)

            p_joints_f = p_joints_f[:,flipidx,:]
            p_joints_f[:,:,0] = im_width - 1 - p_joints_f[:,:,0]

            score_pose_f = joints[:,:,2:] if self.stage_pct == "tokenizer" else \
                encoding_scores_f.mean(1, keepdim=True).repeat(1,p_joints.shape[1],1)

            p_joints = (p_joints + p_joints_f)/2.0
            score_pose = (score_pose + score_pose_f)/2.0
            
        else:
            codebook_indices_f = None
            codebook_distances_f = None

        bbox_ids = None
        if (img_metas is not None) and ('bbox_id' in img_metas[0]):
            bbox_ids = []

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.ones((batch_size, 2), dtype=np.float32)
        score = np.ones(batch_size)
        if img_metas is None:
            c[:, 0] = (im_width-1) / 2
            c[:, 1] = (im_height-1) / 2
            image_paths = None
        else:
            image_paths = []
            for i in range(batch_size):
                c[i, :] = img_metas[i]['center']
                s[i, :] = img_metas[i]['scale']
                image_paths.append(img_metas[i]['image_file'])

                if 'bbox_score' in img_metas[i]:
                    score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
                if bbox_ids is not None:
                    bbox_ids.append(img_metas[i]['bbox_id'])

        p_joints = p_joints.cpu().numpy()
        score_pose = score_pose.cpu().numpy()
        for i in range(p_joints.shape[0]):
            p_joints[i] = transform_preds(
                p_joints[i], c[i], s[i], [im_width, im_height], use_udp=False)
        
        all_preds = np.zeros((batch_size, p_joints.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = p_joints
        all_preds[:, :, 2:3] = score_pose
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        final_preds = {}
        final_preds['preds'] = all_preds
        final_preds['boxes'] = all_boxes
        final_preds['image_paths'] = image_paths
        final_preds['bbox_ids'] = bbox_ids
        final_preds['codebook_indices'] = codebook_indices
        final_preds['codebook_distances'] = codebook_distances
        final_preds['codebook_indices_f'] = codebook_indices_f
        final_preds['codebook_distances_f'] = codebook_distances_f
        results.update(final_preds)
        results['output_heatmap'] = None

        return results

    def show_result(self):
        # Not implemented
        return None
