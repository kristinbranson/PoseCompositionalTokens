# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: VQVAE
#     language: python
#     name: python3
# ---

# %%
# imports 

import sys
print(f'sys.path: {sys.path}')

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
import json
import re
import kristin.kbutils as utils
import time

# so that FlyMABe2022Dataset is registered
from datasets.flymabe_dataset import FlyMABe2022Dataset

# set up environment
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(models.__file__))))
plt.rcParams['savefig.directory'] = os.getcwd()


# %%
#['/groups/branson/home/bransonk/codepacks/PoseCompositionalTokens/kristin', '/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting', '/groups/branson/home/bransonk/codepacks/PoseCompositionalTokens', '/home/bransonk@hhmi.org/miniforge3/envs/VQVAE/lib/python38.zip', '/home/bransonk@hhmi.org/miniforge3/envs/VQVAE/lib/python3.8', '/home/bransonk@hhmi.org/miniforge3/envs/VQVAE/lib/python3.8/lib-dynload', '/home/bransonk@hhmi.org/.local/lib/python3.8/site-packages', '/home/bransonk@hhmi.org/miniforge3/envs/VQVAE/lib/python3.8/site-packages', '/groups/branson/home/bransonk/codepacks/vqtorch', '/home/bransonk@hhmi.org/miniforge3/envs/VQVAE/lib/python3.8/site-packages/setuptools/_vendor']
import sys
print(sys.path)
import vqtorch
print(vqtorch.__file__)


# %%
# parameters

workdir = './work_dirs'
outdir = os.path.join(os.getcwd(),workdir)
configdir = './configs'

configfile = os.path.join(configdir,'flymabe_pkl_pct_ste_sep_tokenizer.py')
cfg = Config.fromfile(configfile)
cfg.work_dir = os.path.join(workdir,os.path.splitext(os.path.basename(configfile))[0])

epoch = 40
checkpoint = os.path.join(cfg.work_dir,f'epoch_{epoch}.pth')

logfile = os.path.join(cfg.work_dir,'20250304_175722.log.json')
logfile = os.path.join(cfg.work_dir,'20250305_100114.log.json')

# print time config files were last modified
print(f'Config {configfile} last modified:',time.ctime(os.path.getmtime(configfile)))

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
eval_batch_size = 8

flyskeledges = utils.get_fly_skeledges_for_plotting(cfg=cfg)
train_imgnorm_info = utils.get_img_norm_info(cfg=cfg,mode='train')
test_imgnorm_info = utils.get_img_norm_info(cfg=cfg,mode='test')



# %%
# load datasets and models

train_dataset = build_dataset(cfg.data.train)
val_dataset = build_dataset(cfg.data.val)
test_dataset = build_dataset(cfg.data.test)

model = build_posenet(cfg.model)
model = model.to(device)

train_dataloader = build_dataloader(train_dataset,
                samples_per_gpu=eval_batch_size,
                workers_per_gpu=1,
                num_gpus=1,
                dist=False,
                shuffle=True,
                seed=None,
                drop_last=False,
                pin_memory=True)


# %% [markdown]
# To train, ran:
# ```
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 ./tools/train.py configs/flymabe_pkl_pct_ste_sep_tokenizer.py --seed=0
# ```

# %%
# load checkpoint
print(f'Checkpoint {checkpoint} last modified:',time.ctime(os.path.getmtime(checkpoint)))
ckpt_info = load_checkpoint(model, checkpoint, map_location=device)

# %%
# evaluate the model

nexamples_plot = 4
assert nexamples_plot <= eval_batch_size

fig,axs = plt.subplots(3,nexamples_plot,figsize=(5*nexamples_plot,15),sharex='row',sharey='row')

ax = axs[0]
exbatch = next(iter(train_dataloader))
with torch.no_grad():
    res = model(img=exbatch['img'].to(device=device),
                joints_3d=exbatch['joints_3d'].to(device=device),
                joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                img_metas=exbatch['img_metas'].data[0],
                return_loss=False)

ax = axs[0,:]
for i in range(nexamples_plot):
    img = utils.unnormalize_image(exbatch['img'][i].numpy().transpose(1,2,0),imgnorm_info=test_imgnorm_info)
    ax[i].imshow(img)
    utils.plot_skeleton(exbatch['joints_3d'][i].numpy(),cfg,
                        joints_visible=exbatch['joints_3d_visible'][i].numpy(),skeledges=flyskeledges,ax=ax[i])
    ax[i].set_title('Test input')
ax = axs[1,:]
for i in range(nexamples_plot):
    img = utils.unnormalize_image(exbatch['img'][i].numpy().transpose(1,2,0),imgnorm_info=test_imgnorm_info)
    ax[i].imshow(img)
    utils.plot_skeleton(res['preds'][i],cfg,skeledges=flyskeledges,ax=ax[i])
    ax[i].set_title('Test output')
    
ax = axs[2,:]
for i in range(nexamples_plot):
    utils.plot_skeleton(exbatch['joints_3d'][i].numpy(),cfg,
                    joints_visible=exbatch['joints_3d_visible'][i].numpy(),
                    skeledges=flyskeledges,ax=ax[i],markeredgecolor=[.5,.5,.5],markerfacecolor=[.5,.5,.5])
    utils.plot_skeleton(res['preds'][i],cfg,skeledges=flyskeledges,ax=ax[i])
    ax[i].set_title('Both')
    ax[i].set_aspect('equal')

ax[0].invert_yaxis()
print('True = gray, pred = color')

# %%
# plot all predicted skeletons on top of each other for batch

fig,ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
for i in range(eval_batch_size):
    utils.plot_skeleton(res['preds'][i],cfg,skeledges=flyskeledges,ax=ax[0])
    utils.plot_skeleton(exbatch['joints_3d'][i].numpy(),cfg,
                        joints_visible=exbatch['joints_3d_visible'][i].numpy(),
                        skeledges=flyskeledges,ax=ax[1])

for i in range(2):
    ax[i].set_aspect('equal')


# %%
# load in loss
loginfo = {'train': {}, 'val': {}}
num_epochs = cfg.total_epochs
with open(logfile) as f:
    while line := f.readline():
        # parse with json
        line = json.loads(line)
        if 'mode' not in line:
            continue
        mode = line['mode']
        epoch = line['epoch']
        ks = line.keys()
        for k in ks:
            if k in ['mode','epoch']:
                continue
            if k not in loginfo[mode]:
                loginfo[mode][k] = np.zeros((num_epochs+1,))
                loginfo[mode][k][:] = np.nan
            loginfo[mode][k][epoch] = line[k]

if 'loss' in loginfo['train']:
    plt.plot(loginfo['train']['loss'])
    plt.xlabel('epoch')
    plt.ylabel('train loss')

# %%
# check if two predictions are the same

exbatch = next(iter(train_dataloader))
with torch.no_grad():
    res = model(img=exbatch['img'].to(device=device),
                joints_3d=exbatch['joints_3d'].to(device=device),
                joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                img_metas=exbatch['img_metas'].data[0],
                return_loss=False)
    
torch.all(res['codebook_indices'][0] == res['codebook_indices'][-1])

# %%
# # count codewords
# unique_codewords = utils.compute_unique_codewords(model,train_dataloader)

# ntokens = cfg.model.keypoint_head.tokenizer.codebook.token_num
# dict_size = cfg.model.keypoint_head.tokenizer.codebook.token_class_num
# if cfg.model.keypoint_head.tokenizer.codebook.share_codebook:
#     all_used_codewords = set().union(*unique_codewords)
#     print(f'total number of codewords used across all tokens: {len(all_used_codewords)} / {dict_size}')
# for tokeni in range(ntokens):
#     print(f'token {tokeni}, total number of unique codewords: {len(unique_codewords[tokeni])}')

# %%
# start training so that we can run debugger
from mmpose.apis import train_model
from mmpose.utils import collect_env

RANDOMSEED = 0

datasets = [train_dataset]
distributed = False
validate = True
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
meta = dict()
# log env info
env_info_dict = collect_env()
env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
meta['env_info'] = env_info
cfg.seed = RANDOMSEED
meta['seed'] = RANDOMSEED

cfg.work_dir = './work_dirs/flymabe_pkl_pct_ste_sep_tokenizer_run2'
cfg.gpu_ids = range(1) 

train_model(
    model,
    datasets,
    cfg,
    distributed=distributed,
    validate=validate,
    timestamp=timestamp,
    meta=meta)
