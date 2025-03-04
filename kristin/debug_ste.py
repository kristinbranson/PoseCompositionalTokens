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
#     display_name: VQVAE
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import models
from mmcv import Config
import copy
from mmpose.datasets import build_dataset, build_dataloader
from models import build_posenet
from mmpose.apis import train_model
from mmcv.runner import load_checkpoint
import kristin.utils as utils

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(models.__file__))))
plt.rcParams['savefig.directory'] = os.getcwd()

inconfigfile = 'configs/fly_pct_ste_sep_tokenizer.py'
cfg = Config.fromfile(inconfigfile)
cfg.gpu_ids = range(1)
cfg.work_dir = os.path.join('./work_dirs',os.path.splitext(os.path.basename(inconfigfile))[0])
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
device = 'cpu'
    
flyskeledges = utils.get_fly_skeledges_for_plotting(cfg=cfg)
train_imgnorm_info = utils.get_img_norm_info(cfg=cfg,mode='train')
test_imgnorm_info = utils.get_img_norm_info(cfg=cfg,mode='test')


# %%
#cfg.model.keypoint_head.tokenizer.codebook.share_codebook = False

# %%
model = build_posenet(cfg.model)
model = model.to(device)
datasets = [build_dataset(cfg.data.train)]

# %%
model.keypoint_head.tokenizer.vqste

# %%
batch_size = 8
dataloader = build_dataloader(datasets[0],
                samples_per_gpu=batch_size,
                workers_per_gpu=1,
                num_gpus=1,
                dist=False,
                shuffle=True,
                seed=None,
                drop_last=False,
                pin_memory=True)
exbatch = next(iter(dataloader))

# %%
with torch.no_grad():

    res = model(img=exbatch['img'].to(device=device),
                joints_3d=exbatch['joints_3d'].to(device=device),
                joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                img_metas=exbatch['img_metas'].data[0],
                return_loss=False)
    
print(res)

# %%
# ran 
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 ./tools/train.py configs/fly_pct_ste_tokenizer.py --launcher pytorch

# %%
# load model checkpoint
checkpoint = 'work_dirs/fly_pct_ste_sep_tokenizer/epoch_200.pth'
load_checkpoint(model, checkpoint, map_location=device)

# %%
# evaluate the model
nexamples_plot = 4
assert nexamples_plot <= batch_size

fig,axs = plt.subplots(3,nexamples_plot,figsize=(5*nexamples_plot,15),sharex='row',sharey='row')

ax = axs[0]
exbatch = next(iter(dataloader))
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
unique_codewords = utils.compute_unique_codewords(model,dataloader)


# %%
ntokens = cfg.model.keypoint_head.tokenizer.codebook.token_num
dict_size = cfg.model.keypoint_head.tokenizer.codebook.token_class_num
all_used_codewords = set().union(*unique_codewords)
print(f'total number of codewords used across all tokens: {len(all_used_codewords)} / {dict_size}')
for tokeni in range(ntokens):
    print(f'token {tokeni}, total number of unique codewords: {len(unique_codewords[tokeni])}')

# %%
for kpinfo in cfg.data.train.dataset_info.keypoint_info.values():
    print(f"'{kpinfo['name']}'")
