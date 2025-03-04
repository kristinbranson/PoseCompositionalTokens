import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
import os
import re
import torch
import tqdm

defaultconfigfile = '/groups/branson/home/bransonk/codepacks/PoseCompositionalTokens/configs/fly_pct_base_noaug_tokenizer.py'

name_translation = {
    'ant_head': ['antennae_midpoint',],
    'left_eye': ['left_eye',],
    'right_eye': ['right_eye',],
    'left_thorax': ['left_front_thorax',],
    'right_thorax': ['right_front_thorax',],
    'pos_notum': ['base_thorax',],
    'pos_abdomen': ['tip_abdomen',],
    'left_front_tar': ['left_front_leg_tip',],
    'right_front_tar': ['right_front_leg_tip',],
    'left_mid_fe': ['left_middle_femur_base',],
    'left_mid_fetib': ['left_middle_femur_tibia_joint',],
    'left_mid_tar': ['left_middle_leg_tip',],
    'right_mid_fe': ['right_middle_femur_base',],
    'right_mid_fetib': ['right_middle_femur_tibia_joint',],
    'right_mid_tar': ['right_middle_leg_tip',],
    'left_back_tar': ['left_back_leg_tip',],
    'right_back_tar': ['right_back_leg_tip',],
    'left_outer_wing': ['left_outer_wing',],
    'left_mid_wing': ['wing_left',],
    'right_outer_wing': ['right_outer_wing',],
    'right_mid_wing': ['wing_right',],
}

def get_kp_info(configfile=None,cfg=None):
    if cfg is None:
        if configfile is None:
            configfile = defaultconfigfile
        cfg = Config.fromfile(configfile)
    return cfg.data.train.dataset_info.keypoint_info

def get_skeleton_info(configfile=None,cfg=None):
    if cfg is None:
        if configfile is None:
            configfile = defaultconfigfile
        cfg = Config.fromfile(configfile)
    return cfg.data.train.dataset_info.skeleton_info

def get_kp_names(configfile=None,cfg=None):
    kpinfo = get_kp_info(configfile=configfile,cfg=cfg)
    nkps = len(kpinfo)
    kpnames = [None,]*nkps
    for k,v in kpinfo.items():
        kpnames[k] = v['name']
    return kpnames

def translate_kp_names(kpnames):
    kpnames_trans = [None,]*len(kpnames)
    for k,v in name_translation.items():
        if k in kpnames:
            i = kpnames.index(k)
            kpnames_trans[i] = k
        else:
            for v2 in v:
                if v2 in kpnames:
                    i = kpnames.index(v2)
                    kpnames_trans[i] = k
                    break
                
    return kpnames_trans

def get_fly_skeledges_for_plotting(kpinfo=None,configfile=None,cfg=None):
    if kpinfo is None:
        kpinfo = get_kp_info(configfile=configfile,cfg=cfg)
    kpnames = get_kp_names(configfile=configfile,cfg=cfg)
    kpnames0 = translate_kp_names(kpnames)
    flyskeledges_names = [
        ('ant_head','right_eye'),
        ('ant_head','left_eye'),
        ('left_eye','right_eye'),
        ('left_thorax','right_thorax'),
        ('pos_notum','left_thorax'),
        ('pos_notum','right_thorax'),
        ('pos_notum','pos_abdomen'),
        ('left_thorax','left_front_tar'),
        ('right_thorax','right_front_tar'),
        ('pos_notum','left_mid_fe'),
        ('pos_notum','right_mid_fe'),
        ('left_mid_fe','left_mid_fetib'),
        ('right_mid_fe','right_mid_fetib'),
        ('left_mid_fetib','left_mid_tar'),
        ('right_mid_fetib','right_mid_tar'),
        ('pos_notum','left_back_tar'),
        ('pos_notum','right_back_tar'),
        ('pos_notum','left_mid_wing'),
        ('left_mid_wing','left_outer_wing'),
        ('pos_notum','right_mid_wing'),
        ('right_mid_wing','right_outer_wing'),
    ]

    flyskeledges = {}
    for i, (n1,n2) in enumerate(flyskeledges_names):
        i1 = kpnames0.index(n1)
        i2 = kpnames0.index(n2)
        color = kpinfo[i2]['color']
        flyskeledges[i] = {'link': (kpnames[i1],kpnames[i2]), 'color': color}
        
    return flyskeledges

def plot_skeleton(joints,cfg,joints_visible=None,ax=None,edgecolor=None,markeredgecolor='k',
                  markerfacecolor=None,skeledges=None,hplot=None):
    if hplot is None:
        hplot = {}
    kpnames = [v['name'] for v in cfg['dataset_info']['keypoint_info'].values()]
    if ax is None:
        if 'ax' in hplot:
            ax = hplot['ax']
        else:
            ax = plt.gca()
    if skeledges is None:
        skeledges = cfg['dataset_info']['skeleton_info']

    hplot['ax'] = ax
    
    if 'edges' not in hplot:
        hplot['edges'] = []
    if 'pts' not in hplot:
        hplot['pts'] = []
        
    for i,edge in skeledges.items():
        kp1 = edge['link'][0]
        kp2 = edge['link'][1]
        kpi1 = kpnames.index(kp1)
        kpi2 = kpnames.index(kp2)
        x = joints[[kpi1,kpi2],0]
        y = joints[[kpi1,kpi2],1]
        if joints_visible is not None:
            x[joints_visible[kpi1][0] == 0] = np.nan
            y[joints_visible[kpi2][0] == 0] = np.nan
        if len(hplot['edges']) <= i:
            if edgecolor is None:
                c = edge['color']
            else:
                c = edgecolor
            hplot['edges'] += ax.plot(x,y,'-',color=c)
        else:
            hplot['edges'][i].set_xdata(x)
            hplot['edges'][i].set_ydata(y)
            
    for kpi,kpinfo in cfg['dataset_info']['keypoint_info'].items():   
        x = joints[kpi,0]
        y = joints[kpi,1]
        if joints_visible is not None:
            if joints_visible[kpi][0] == 0:
                x = np.nan
                y = np.nan
        if len(hplot['pts']) <= kpi:
            if markerfacecolor is None:
                c = kpinfo['color']
            else:
                c = markerfacecolor
            hplot['pts'] += ax.plot(x,y,'o',color=c,markeredgecolor=markeredgecolor)
        else:
            hplot['pts'][kpi].set_xdata(x)
            hplot['pts'][kpi].set_ydata(y)
    return hplot

def get_img_norm_info(cfg,mode):
    assert mode in ['train','val','test']
    transforms = cfg[mode+'_pipeline']
    for t in transforms:
        if t['type'] == 'NormalizeTensor':
            return {'mean': np.array(t['mean']), 'std': np.array(t['std'])}
    return None

def unnormalize_image(img,imgnorm_info=None,cfg=None,mode=None):
    if imgnorm_info is None:
        imgnorm_info = get_img_norm_info(cfg,mode)
    img_unnorm = img.copy() * imgnorm_info['std'] + imgnorm_info['mean']
    return img_unnorm

def configfile_to_outdir(configfile,workdir):
    outdir = os.path.join(workdir,os.path.basename(configfile).replace('.py',''))
    return outdir
    
def get_last_checkpoint(configfile,workdir):
    
    # find all files named 'epoch_*.pth' in outdir and parse out the epoch number
    outdir = configfile_to_outdir(configfile,workdir)
    files = os.listdir(outdir)
    pattern = re.compile(r'epoch_(\d+)\.pth')
    max_epoch = None
    checkpoint = None
    for file in files:
        match = pattern.match(str(file))
        if match:
            epoch_num = int(match.group(1))
            if max_epoch is None or epoch_num > max_epoch:
                max_epoch = epoch_num
                checkpoint = os.path.join(outdir,file)
    return checkpoint,max_epoch

def dict_val_to_key(d,v):
    for k,v2 in d.items():
        if v2 == v:
            return k
    return None

def compute_unique_codewords(model,dataloader):
    token_num = model.keypoint_head.tokenizer.token_num
    dictsize = model.keypoint_head.tokenizer.token_class_num
    
    device = next(model.parameters()).device
    
    # what can the encoder produce
    unique_codewords = [set() for _ in range(token_num)]
    for batch in tqdm.tqdm(dataloader):
        with torch.no_grad():
            res = model(img=batch['img'].to(device=device),
                        joints_3d=batch['joints_3d'].to(device=device),
                        joints_3d_visible=batch['joints_3d_visible'].to(device=device),
                        img_metas=batch['img_metas'].data[0],
                        return_loss=False)
        
            codewords = res['codebook_indices'].cpu().numpy()
            for tokeni in range(token_num):
                unique_codewords[tokeni].update(codewords[:,tokeni])
                
    return unique_codewords