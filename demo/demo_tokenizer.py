# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: PCT
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import models
from models import build_posenet
from mmcv import Config
from mmpose.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(models.__file__))))
configfile = 'configs/pct_base_woimgguide_tokenizer.py'
checkpoint = 'weights/tokenizer/swin_base_woimgguide.pth'
plt.rcParams['savefig.directory'] = os.getcwd()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# %%

# # copy in plotting from demo_img_with_mmdet.py
# made some minor updates so that it doesn't plot missing joints

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.link_pairs)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))
            
            
color2 = [(252,176,243),(252,176,243),(252,176,243),
    (0,176,240), (0,176,240), (0,176,240),
    (255,255,0), (255,255,0),(169, 209, 142),
    (169, 209, 142),(169, 209, 142),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127)]

link_pairs2 = [
        [15, 13], [13, 11], [11, 5], 
        [12, 14], [14, 16], [12, 6], 
        [9, 7], [7,5], [5, 6], [6, 8], [8, 10],
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        ]


point_color2 = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)


def map_joint_dict(joints,joints_visible):
    joints_dict = {}
    for i in range(joints.shape[0]):
        if (joints_visible is not None) and (joints_visible[i][0] == 0):
            x = np.nan
            y = np.nan
        else:
            x = joints[i][0]
            y = joints[i][1]
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict

def plot_skeleton(dt_joints,dt_joints_visible=None,ax=None,thickness=1):
    
    if ax is None:
        ax = plt.gca()
        
    joints_dict = map_joint_dict(dt_joints,dt_joints_visible)
    
    # stick 
    for k, link_pair in enumerate(chunhua_style.link_pairs):
        if k in range(11,16):
            lw = thickness
        else:
            lw = thickness * 2

        line = mlines.Line2D(
                np.array([joints_dict[link_pair[0]][0],
                            joints_dict[link_pair[1]][0]]),
                np.array([joints_dict[link_pair[0]][1],
                            joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=link_pair[2],)
        line.set_zorder(0)
        ax.add_line(line)

    # black ring
    for k in range(dt_joints.shape[0]):
        if k in range(5):
            radius = thickness
        else:
            radius = thickness * 2

        circle = mpatches.Circle(tuple(joints_dict[k]), 
                                    radius=radius, 
                                    ec='black', 
                                    fc=chunhua_style.ring_color[k], 
                                    alpha=1, 
                                    linewidth=1)
        circle.set_zorder(1)
        ax.add_patch(circle)


# %%
# load data

# read the configfile
cfg = Config.fromfile(configfile)

# create the dataset
dataset = build_dataset(cfg.data.test)

# %%
# plot an example to test dataset
exi = np.random.randint(0,len(dataset))
ex = dataset[exi]
im = ex['img'].permute(1, 2, 0)
print(ex.keys())
print(ex['img_metas'].data.keys())
print(ex['img_metas'].data['center'])
imraw = cv2.imread(ex['img_metas'].data['image_file'])
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(im/5+.5)
ax[0].axis('tight')
plot_skeleton(ex['joints_3d'],dt_joints_visible=ex['joints_3d_visible'],ax=ax[0])
ax[0].set_title('transformed')
ax[0].plot(ex['joints_3d'][:,0],ex['joints_3d'][:,1],'r.')
ax[1].imshow(imraw[:,:,::-1]) 
ax[1].plot(ex['img_metas'].data['center'][0],ex['img_metas'].data['center'][1],'ro')
ax[1].set_title('raw')

# %%
# load model
model = build_posenet(cfg.model)
model = model.to(device)
if checkpoint is not None:
    # load model checkpoint
    load_checkpoint(model, checkpoint, map_location=device)

# %%
# create dataloader. i didn't use the dataloader parameters from the config file; batch size = 1
dataloader = build_dataloader(dataset,
                samples_per_gpu=1,
                workers_per_gpu=1,
                num_gpus=1,
                dist=False,
                shuffle=False,
                seed=None,
                drop_last=False,
                pin_memory=True)

# get an example batch 
exbatch = next(iter(dataloader))
for k,v in exbatch.items():
    if type(v) == torch.Tensor:
        print(f'{k}: {v.shape}')
    else:
        print(f'{k}: {type(v)}')

# %%
# run the model on the example batch (batch size = 1)
# return_loss = False is important; this tells it to return the predictions rather than the loss
# the loaded model as defined in the configfile should have model.stage_pct == 'tokenizer'
# I tested with the woimguide model, so it shouldn't actually be using the image input, but it does require and manipulate the input anyways...

print('Evaluate model')

with torch.no_grad():
    res = model(img=exbatch['img'].to(device=device),
                joints_3d=exbatch['joints_3d'].to(device=device),
                joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                img_metas=exbatch['img_metas'].data[0],
                return_loss=False)

# %%
# try setting the image to 0 and see if it affects thre sults
img_black = exbatch['img'].clone()
img_black[:] = 0
with torch.no_grad():
    res_black = model(img=img_black.to(device=device),
                    joints_3d=exbatch['joints_3d'].to(device=device),
                    joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                    img_metas=exbatch['img_metas'].data[0],
                    return_loss=False)

# %%
# no image
with torch.no_grad():
    res_none = model(img=None,
                    joints_3d=exbatch['joints_3d'].to(device=device),
                    joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                    img_metas=None,
                    return_loss=False)

# %%
# plot output
ex = {}
for k,v in exbatch.items():
    if type(v) == torch.Tensor:
        ex[k] = v[0].cpu().numpy()

im = ex['img'].transpose(1, 2, 0)
imraw = cv2.imread(exbatch['img_metas'].data[0][0]['image_file'])

fig,ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(im/5+.5)
plot_skeleton(ex['joints_3d'],dt_joints_visible=ex['joints_3d_visible'],ax=ax[0])
ax[0].axis('tight')
ax[0].axis('image')
ax[0].set_title('input')
ax[1].imshow(imraw[:,:,::-1])
plot_skeleton(res['preds'][0],ax=ax[1])
ax[1].axis('tight')
ax[1].axis('image')
ax[1].set_title('output')
ax[2].imshow(imraw[:,:,::-1])
plot_skeleton(res_black['preds'][0],ax=ax[2])
ax[2].axis('tight')
ax[2].axis('image')
ax[2].set_title('output with black image')
fig.tight_layout()
plt.show()

# %%
fig,ax = plt.subplots(1,3,figsize=(10,5))
plot_skeleton(exbatch['joints_3d'][0],dt_joints_visible=exbatch['joints_3d_visible'][0],ax=ax[0])
ax[0].set_title('input')
plot_skeleton(res['preds'][0],ax=ax[1])
ax[1].set_title('output')
plot_skeleton(res_none['preds'][0],ax=ax[2])
ax[2].set_title('output with no image')
for a in ax:
    a.axis('image')
    a.invert_yaxis()
fig.tight_layout()

# %%
# test that forward_train works
with torch.no_grad():
    res_train = model(img=None,
                    joints_3d=exbatch['joints_3d'].to(device=device),
                    joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                    img_metas=None,
                    return_loss=True)

# %%
# try flies pretending to be people

flyconfigfile = 'configs/dummy_fly_pct_base_woimgguide_tokenizer.py'
flycfg = Config.fromfile(flyconfigfile).data.test

# create the dataset
flydataset = build_dataset(flycfg)

flydataloader = build_dataloader(flydataset,
                                samples_per_gpu=4,
                                workers_per_gpu=1,
                                num_gpus=1,
                                dist=False,
                                shuffle=False,
                                seed=None,
                                drop_last=False,
                                pin_memory=True)

# get an example batch 
exbatch = next(iter(flydataloader))
for k,v in exbatch.items():
    if type(v) == torch.Tensor:
        print(f'{k}: {v.shape}')
    else:
        print(f'{k}: {type(v)}')

# %%
with torch.no_grad():
    flyres = model(img=None,
                joints_3d=exbatch['joints_3d'].to(device=device),
                joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                img_metas=None,
                return_loss=False)


# %%
def plot_skeleton_kb(joints,cfg,joints_visible=None,ax=None):
    kpnames = [v['name'] for v in cfg['dataset_info']['keypoint_info'].values()]
    if ax is None:
        ax = plt.gca()
    for i,edge in cfg['dataset_info']['skeleton_info'].items():
        kp1 = edge['link'][0]
        kp2 = edge['link'][1]
        kpi1 = kpnames.index(kp1)
        kpi2 = kpnames.index(kp2)
        if joints_visible is not None:
            if joints_visible[kpi1][0] == 0 or joints_visible[kpi2][0] == 0:
                continue
        c = edge['color']
        ax.plot(joints[[kpi1,kpi2],0],joints[[kpi1,kpi2],1],'-',color=c)
    for kpi,kpinfo in cfg['dataset_info']['keypoint_info'].items():   
        if joints_visible is not None:
            if joints_visible[kpi][0] == 0:
                continue
        c = kpinfo['color']
        ax.plot(joints[kpi,0],joints[kpi,1],'o',color=c,markeredgecolor='black')



# %%
fig,ax = plt.subplots(exbatch['joints_3d'].shape[0],2,figsize=(10,5*exbatch['joints_3d'].shape[0]))
for i in range(exbatch['joints_3d'].shape[0]):
    plot_skeleton_kb(exbatch['joints_3d'][i],flycfg,joints_visible=exbatch['joints_3d_visible'][i],ax=ax[i,0])
    ax[i,0].imshow(exbatch['img'][i].permute(1,2,0)/5+.5)
    ax[i,0].set_title('fly input')
    plot_skeleton_kb(flyres['preds'][i],flycfg,ax=ax[i,1])
    ax[i,1].set_title('fly output')
    ax[i,1].invert_yaxis()
for a in ax.flatten():
    a.axis('equal')
    a.axis('image')

fig.tight_layout()

# %%
# test that forward_train works
with torch.no_grad():
    res_train = model(img=None,
                    joints_3d=exbatch['joints_3d'].to(device=device),
                    joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                    img_metas=None,
                    return_loss=True)
print(res_train)

# %%
# try config v2 that doesn't load the images

flyconfigfile = 'configs/dummy_fly_pct_base_woimgguide_tokenizer_v2.py'
flycfg = Config.fromfile(flyconfigfile).data.test

# create the dataset
flydataset = build_dataset(flycfg)
flydataset[0]

flydataloader = build_dataloader(flydataset,
                                samples_per_gpu=4,
                                workers_per_gpu=1,
                                num_gpus=1,
                                dist=False,
                                shuffle=True,
                                seed=None,
                                drop_last=False,
                                pin_memory=True)

# get an example batch 
exbatch = next(iter(flydataloader))
for k,v in exbatch.items():
    if type(v) == torch.Tensor:
        print(f'{k}: {v.shape}')
    else:
        print(f'{k}: {type(v)}')

# %%
with torch.no_grad():
    flyres = model(img=None,
                joints_3d=exbatch['joints_3d'].to(device=device),
                joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                img_metas=None,
                return_loss=False)

fig,ax = plt.subplots(exbatch['joints_3d'].shape[0],2,figsize=(10,5*exbatch['joints_3d'].shape[0]))
for i in range(exbatch['joints_3d'].shape[0]):
    plot_skeleton_kb(exbatch['joints_3d'][i],flycfg,joints_visible=exbatch['joints_3d_visible'][i],ax=ax[i,0])
    ax[i,0].set_title('fly input')
    plot_skeleton_kb(flyres['preds'][i],flycfg,ax=ax[i,1])
    ax[i,1].set_title('fly output')
for a in ax.flatten():
    a.invert_yaxis()
    a.axis('equal')
    a.axis('image')

fig.tight_layout()

# %%
with torch.no_grad():
    flyrestrain = model(img=None,
                        joints_3d=exbatch['joints_3d'].to(device=device),
                        joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                        img_metas=None,
                        return_loss=True)
print('successfully predicted with return_loss=True')
print(flyrestrain)

# %%
# the training pipeline unfortunately requires an image, also if we want data augmentation on the pose, it requires an image
# for ease of programming, let's just leave the image in there

# try config file with all keypoints

flyconfigfile = 'configs/fly_pct_base_woimgguide_tokenizer.py'
flycfg = Config.fromfile(flyconfigfile)

# create the dataset
flydataset = build_dataset(flycfg.data.train)
flydatasettest = build_dataset(flycfg.data.test)

flydataloader = build_dataloader(flydataset,
                                samples_per_gpu=4,
                                workers_per_gpu=1,
                                num_gpus=1,
                                dist=False,
                                shuffle=True,
                                seed=None,
                                drop_last=False,
                                pin_memory=True)
flydataloadertest = build_dataloader(flydatasettest,
                                    samples_per_gpu=4,
                                    workers_per_gpu=1,
                                    num_gpus=1,
                                    dist=False,
                                    shuffle=True,
                                    seed=None,
                                    drop_last=False,
                                    pin_memory=True)

# get an example batch 
exbatch = next(iter(flydataloader))
exbatchtest = next(iter(flydataloadertest))
for k,v in exbatch.items():
    if type(v) == torch.Tensor:
        print(f'{k}: {v.shape}')
    else:
        print(f'{k}: {type(v)}')

# %%
fig,ax = plt.subplots(2,exbatch['joints_3d'].shape[0],figsize=(5*exbatch['joints_3d'].shape[0],10))
for i in range(exbatch['joints_3d'].shape[0]):
    ax[0,i].imshow(exbatch['img'][i].permute(1,2,0)/5+.5)
    plot_skeleton_kb(exbatch['joints_3d'][i],flycfg,joints_visible=exbatch['joints_3d_visible'][i],ax=ax[0,i])
    ax[0,i].set_title('train')
    ax[0,i].axis('image')
for i in range(exbatchtest['joints_3d'].shape[0]):
    ax[1,i].imshow(exbatchtest['img'][i].permute(1,2,0)/5+.5)
    plot_skeleton_kb(exbatchtest['joints_3d'][i],flycfg,joints_visible=exbatchtest['joints_3d_visible'][i],ax=ax[1,i])
    ax[1,i].set_title('test')
    ax[1,i].axis('image')

fig.tight_layout()

# %%
import re
# load model
# last file named epoch_*.pth 
flycheckpointdir = 'work_dirs/fly_pct_base_woimgguide_tokenizer'
flycheckpoints = os.listdir(flycheckpointdir)
lastepoch = -1
for f in flycheckpoints:
    if f.startswith('epoch_'):
        # use regular expression to parse the epoch number
        epoch = re.search('epoch_(\d+).pth',f)
        if epoch is None:
            continue
        epoch = int(epoch.group(1))
        if epoch > lastepoch:
            lastepoch = epoch
            flycheckpoint = os.path.join(flycheckpointdir,f)
print(flycheckpoint)
#flycheckpoint = 'work_dirs/fly_pct_base_woimgguide_tokenizer/epoch_10.pth'
flymodel = build_posenet(flycfg.model)
flymodel = flymodel.to(device)
if flycheckpoint is not None:
    # load model checkpoint
    load_checkpoint(flymodel, flycheckpoint, map_location=device)
    
with torch.no_grad():
    flyres = flymodel(img=None,
                    joints_3d=exbatchtest['joints_3d'].to(device=device),
                    joints_3d_visible=exbatchtest['joints_3d_visible'].to(device=device),
                    img_metas=None,
                    return_loss=False)

# %%
fig,ax = plt.subplots(exbatchtest['joints_3d'].shape[0],2,figsize=(10,5*exbatchtest['joints_3d'].shape[0]))
for i in range(exbatchtest['joints_3d'].shape[0]):
    ax[i,0].imshow(exbatchtest['img'][i].permute(1,2,0)/5+.5)
    plot_skeleton_kb(exbatchtest['joints_3d'][i],flycfg,joints_visible=exbatchtest['joints_3d_visible'][i],ax=ax[i,0])
    ax[i,0].set_title('fly input')
    #ax[i,1].imshow(exbatchtest['img'][i].permute(1,2,0)/5+.5)
    plot_skeleton_kb(flyres['preds'][i],flycfg,ax=ax[i,1])
    ax[i,1].set_title('fly output')
    ax[i,1].invert_yaxis()
for a in ax.flatten():
    a.axis('equal')
    a.axis('image')

fig.tight_layout()
