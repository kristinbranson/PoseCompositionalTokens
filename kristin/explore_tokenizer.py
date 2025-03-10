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

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(models.__file__))))
configfile = 'configs/pct_base_woimgguide_tokenizer.py'
configfile = 'configs/flymabe_pkl_pct_ste_sep_tokenizer.py'
cfg = Config.fromfile(configfile)
cfg.work_dir = os.path.join('./work_dirs',os.path.splitext(os.path.basename(inconfigfile))[0])
epoch = 135
checkpoint = os.path.join(cfg.work_dir,f'epoch_{epoch}.pth')
assert os.path.exists(checkpoint)
# print time checkpoint and config files were last modified
print('Checkpoint last modified:',time.ctime(os.path.getmtime(checkpoint)))
print('Config last modified:',time.ctime(os.path.getmtime(configfile)))


plt.rcParams['savefig.directory'] = os.getcwd()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
outdir = os.path.join(os.getcwd(),'work_dirs')
configdir = 'configs'


# %%
# find all files named 'fly_pct_base_ntokens{token_num}_dictsize{dict_size}_tokenizer.py' 
# in configdir and parse out token_num and dict_size
def find_config_files(configdir):
    pattern = re.compile(r'fly_pct_base_ntokens(\d+)_dictsize(\d+)_tokenizer\.py$')
    configfiles = {}
    for filepath in os.listdir(configdir):
        match = pattern.search(str(filepath))
        if match:
            token_num = int(match.group(1))
            dict_size = int(match.group(2))
            configfiles[(token_num, dict_size)] = os.path.join(configdir,filepath)
    return configfiles

configfiles = find_config_files(configdir)
token_nums = sorted(set([key[0] for key in configfiles.keys()]))
dict_sizes = sorted(set([key[1] for key in configfiles.keys()]))
print(f'token_nums = {token_nums}')
print(f'dict_sizes = {dict_sizes}')

# %%
# choose one config
token_num = 16
dict_size = 256
batch_size = 8

configfile = configfiles[(token_num, dict_size)]

# read the configfile
cfg = Config.fromfile(configfile)

# create the datasets
train_dataset = build_dataset(cfg.data.train)
test_dataset = build_dataset(cfg.data.test)

# create dataloader. i didn't use the dataloader parameters from the config file
train_dataloader = build_dataloader(train_dataset,
                                    samples_per_gpu=batch_size,
                                    workers_per_gpu=1,
                                    num_gpus=1,
                                    dist=False,
                                    shuffle=False,
                                    seed=None,
                                    drop_last=False,
                                    pin_memory=True)

test_dataloader = build_dataloader(test_dataset,
                                    samples_per_gpu=batch_size,
                                    workers_per_gpu=1,
                                    num_gpus=1,
                                    dist=False,
                                    shuffle=False,
                                    seed=None,
                                    drop_last=False,
                                    pin_memory=True)



# plot some examples
fig,ax = plt.subplots(1,2,figsize=(10,5))

flyskeledges = utils.get_fly_skeledges_for_plotting(cfg=cfg)
train_imgnorm_info = utils.get_img_norm_info(cfg=cfg,mode='train')
test_imgnorm_info = utils.get_img_norm_info(cfg=cfg,mode='test')

ex = test_dataset[0]
img = utils.unnormalize_image(ex['img'].numpy().transpose(1,2,0),imgnorm_info=test_imgnorm_info)
ax[0].imshow(img)
utils.plot_skeleton(ex['joints_3d'],cfg,joints_visible=ex['joints_3d_visible'],skeledges=flyskeledges,ax=ax[0])
ax[0].set_title('Test')
ex = train_dataset[0]
img = utils.unnormalize_image(ex['img'].numpy().transpose(1,2,0),imgnorm_info=train_imgnorm_info)
ax[1].imshow(img)
utils.plot_skeleton(ex['joints_3d'],cfg,joints_visible=ex['joints_3d_visible'],skeledges=flyskeledges,ax=ax[1])
ax[1].set_title('Train')


# %%
# load in the model
checkpoint,epoch = utils.get_last_checkpoint(configfile,outdir)

# load model
model = build_posenet(cfg.model)
model = model.to(device)
if checkpoint is not None:
    # load model checkpoint
    load_checkpoint(model, checkpoint, map_location=device)

# %%
# evaluate the model
nexamples_plot = 4
assert nexamples_plot <= batch_size

fig,axs = plt.subplots(3,nexamples_plot,figsize=(5*nexamples_plot,15),sharex='row',sharey='row')

ax = axs[0]
exbatch = next(iter(test_dataloader))
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
    print(f'Example {i}: codebook indices = {res["codebook_indices"][i].cpu().numpy()}')
    
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
# compute the mean over all training examples
meanfly = 0
n = 0
for ex in train_dataset:
    meanfly += ex['joints_3d']*ex['joints_3d_visible']
    n += ex['joints_3d_visible']
n[n==0] = 1
meanfly /= n

utils.plot_skeleton(meanfly,cfg,skeledges=flyskeledges)
plt.gca().invert_yaxis()
plt.title('Mean fly')
plt.axis('equal')

# %%
baseex = {'img': None, 
        'joints_3d': torch.tensor(meanfly[None]), 
        'joints_3d_visible': torch.ones(1,meanfly.shape[0]),
        'img_metas': ex['img_metas']}

with torch.no_grad():
    res = model(img=None,
        joints_3d=baseex['joints_3d'].to(device=device),
        joints_3d_visible=baseex['joints_3d_visible'].to(device=device),
        img_metas=None,
        return_loss=False)
codebook_base = res['codebook_indices']


# %%
def order_codebook(p,tokeni):
    dict_size = p.shape[0]
    i0 = codebook_base[0,tokeni].item()
    isleft = np.ones(dict_size,dtype=bool)
    isleft[i0] = False
    codebook_order = np.zeros(dict_size,dtype=int)
    codebook_order[0] = i0

    dtotal = 0
    dpath = np.zeros(dict_size)
    dpath[:] = np.nan
    p = p.reshape(dict_size,-1)
    for i in range(dict_size-1):
        d = np.linalg.norm(p[isleft]-p[i0],axis=1)
        j = np.argmin(d)
        dtotal += d[j]
        dpath[i+1] = d[j]
        # get index of j
        i0 = np.where(isleft)[0][j]
        isleft[i0] = False
        codebook_order[i+1] = i0
    return codebook_order,dpath,dtotal

def visualize_token(tokeni,codebook_base,dict_size,model,device):
    codebook_curr = codebook_base.clone().to(device=device)
    with torch.no_grad():
        for i in range(dict_size):
            codebook_curr[:,tokeni] = i
            p_joints_curr = model.keypoint_head.tokenizer.decode_codewords(encoding_indices=codebook_curr)[0]
            if i == 0:
                p_joints = torch.zeros((dict_size,)+p_joints_curr.shape,device=device)
            p_joints[i] = p_joints_curr

    p_joints = p_joints.cpu().numpy()
    codebook_order,dpath,dtotal = order_codebook(p_joints,tokeni)

    return p_joints,codebook_order,dpath,dtotal



# %%
# # do pca on p_joints where an example is a row
# p_joints_flat = p_joints.reshape((dict_size,-1))
# mean = np.mean(p_joints_flat,axis=0)
# p_joints_flat = p_joints_flat - mean
# cov = np.cov(p_joints_flat,rowvar=False)
# evals,evecs = np.linalg.eig(cov)
# idx = np.argsort(evals)[::-1]
# evecs = evecs[:,idx]
# evals = evals[idx]
# p_joints_pca = np.dot(p_joints_flat,evecs[:,:2])
# plt.plot(evals,'.-')

# %%
# make an animation that plots the skeleton of the fly for each p_joints output
# from the codebook
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# skip codebooks with distance 0
dthresh = 0

def show_token_animation(p_joints,codebook_order_nz,cfg,flyskeledges,ax=None,naxr=None,naxc=None,tokensplot=None):

    if type(p_joints) is not list:
        p_joints = [p_joints,]
    if type(codebook_order_nz) is not list:
        codebook_order_nz = [codebook_order_nz,]
    if tokensplot is not None and not hasattr(tokensplot,'__len__'):
        tokensplot = [tokensplot,]

    ntokensplot = len(p_joints)
    if naxr is None and naxc is None:
        naxc = int(np.ceil(np.sqrt(ntokensplot)))
        naxr = int(np.ceil(ntokensplot/naxc))
    elif naxr is None:
        naxr = int(np.ceil(ntokensplot/naxc))
    elif naxc is None:
        naxc = int(np.ceil(ntokensplot/naxr))
    assert naxr*naxc >= ntokensplot
    
    minp = np.min([np.min(p,axis=(0,1)) for p in p_joints],axis=0)
    maxp = np.max([np.max(p,axis=(0,1)) for p in p_joints],axis=0)
    dp = maxp - minp
    border = .05
    xlim = [minp[0]-border*dp[0],maxp[0]+border*dp[0]]
    ylim = [minp[1]-border*dp[1],maxp[1]+border*dp[1]]

    if ax is None:
        
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        if dx > dy:
            axsizex = 5
            axsizey = axsizex * dy/dx
        else:
            axsizey = 5
            axsizex = axsizey * dx/dy
        
        fig,ax = plt.subplots(naxr,naxc,sharex=True,sharey=True,figsize=(axsizex*naxc,axsizey*naxr),squeeze=False)
        ax = ax.flatten()
    else:
        # make sure ax is a 1d array
        ax = np.array(ax).flatten()
        fig = ax[0].figure
        assert len(ax) >= ntokensplot
    
    hplots = []
    for tokeni in range(ntokensplot):
        hplot = utils.plot_skeleton(p_joints[tokeni][codebook_order_nz[tokeni][0]],cfg,skeledges=flyskeledges,ax=ax[tokeni])
        hplots.append(hplot)
        ax[tokeni].set_aspect('equal')
        s = f'cw = {codebook_order_nz[tokeni][0]}'
        if tokensplot is not None:
            s = f'Token {tokensplot[tokeni]}, '+s
        ax[tokeni].set_title(s)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].invert_yaxis()
    
    fig.tight_layout()
    
    ncodewords = [len(x) for x in codebook_order_nz]
    maxncodewords = max(ncodewords)
        
    def animate(i):
        for tokeni in range(ntokensplot):
            j = i % ncodewords[tokeni]
            utils.plot_skeleton(p_joints[tokeni][codebook_order_nz[tokeni][j]],cfg,skeledges=flyskeledges,ax=ax[tokeni],hplot=hplots[tokeni])
            s = f'cw = {codebook_order_nz[tokeni][j]}'
            if tokensplot is not None:
                s = f'Token {tokensplot[tokeni]}, '+s
            ax[tokeni].set_title(s)


    ani = animation.FuncAnimation(fig, animate, frames=maxncodewords, interval=100)
    return ani

# %%
# show one token from mean base

tokeni = 1
p_joints,codebook_order,dpath,dtotal = visualize_token(tokeni,codebook_base,dict_size,model,device)
plt.plot(dpath)

codebook_order_nz = codebook_order[np.isnan(dpath) | (dpath > dthresh)]

ani = show_token_animation(p_joints,codebook_order_nz,cfg,flyskeledges,tokensplot=tokeni)
from IPython.display import HTML
HTML(ani.to_jshtml())


# %%
# show all tokens from mean base
ntokensplot = token_num
if ntokensplot < token_num:
    tokensplot = np.random.choice(token_num-1,ntokensplot,replace=False)
else:
    tokensplot = np.arange(token_num,dtype=int)
print(tokensplot)
p_joints = []
codebook_order_nz = []
for tokencurr in tokensplot:
    p_joints_curr,codebook_order_curr,dpath_curr,dtotal_curr = visualize_token(tokencurr,codebook_base,dict_size,model,device)
    p_joints.append(p_joints_curr)
    codebook_order_nz_curr = codebook_order_curr[np.isnan(dpath_curr) | (dpath_curr>dthresh)]
    codebook_order_nz.append(codebook_order_nz_curr)
ani = show_token_animation(p_joints,codebook_order_nz,cfg,flyskeledges,tokensplot=tokensplot)

# write animation to video 
outfigdir = os.path.join(outdir,'figs')
os.makedirs(outfigdir,exist_ok=True)
outvidfile = os.path.join(outfigdir,f'tokenvis_basemean_ntokens{token_num}_dictsize{dict_size}.mp4')
# use pillow writer to write mp4
ani.save(outvidfile, writer='ffmpeg', fps=10)

# %%
# choose some base samples for each behavior type
testann = json.load(open(cfg.data.test.ann_file))
nbasepercat = 5
behaviors = testann['info']['behaviors']
print(behaviors)

for behkey,behname in behaviors.items():
    exidx = np.array([i for i,ex in enumerate(testann['annotations']) if ex['behavior'] == int(behkey)])
    print(f'n {behname} test examples: {len(exidx)}')

    for exii in range(nbasepercat):

        exi = np.random.choice(exidx)
        behex = test_dataset[exi]
        with torch.no_grad():
            res = model(img=None,
                joints_3d=torch.tensor(behex['joints_3d'][None]).to(device=device),
                joints_3d_visible=torch.tensor(behex['joints_3d_visible'][None]).to(device=device),
                img_metas=None,
                return_loss=False)
        codebook_beh_base = res['codebook_indices']

        # show all tokens
        ntokensplot = token_num
        if ntokensplot < token_num:
            tokensplot = np.random.choice(token_num-1,ntokensplot,replace=False)
        else:
            tokensplot = np.arange(token_num,dtype=int)
        p_joints = []
        codebook_order_nz = []
        for tokencurr in tokensplot:
            p_joints_curr,codebook_order_curr,dpath_curr,dtotal_curr = visualize_token(tokencurr,codebook_beh_base,dict_size,model,device)
            p_joints.append(p_joints_curr)
            codebook_order_nz_curr = codebook_order_curr[np.isnan(dpath_curr) | (dpath_curr>dthresh)]
            codebook_order_nz.append(codebook_order_nz_curr)
        ani = show_token_animation(p_joints,codebook_order_nz,cfg,flyskeledges,tokensplot=tokensplot)

        # write animation to video 
        outfigdir = os.path.join(outdir,'figs')
        os.makedirs(outfigdir,exist_ok=True)
        outvidfile = os.path.join(outfigdir,f'tokenvis_base{behname}{exi}_ntokens{token_num}_dictsize{dict_size}.mp4')
        print(f'Saving example for behavior {behname} example {exi} ({exii}/{nbasepercat}) to {outvidfile}')
        # use pillow writer to write mp4
        ani.save(outvidfile, writer='ffmpeg', fps=10)
        
        plt.close('all')
        


# %%
baseex = {'img': None, 
        'joints_3d': torch.tensor(meanfly[None]), 
        'joints_3d_visible': torch.ones(1,meanfly.shape[0]),
        'img_metas': ex['img_metas']}

with torch.no_grad():
    res = model(img=None,
        joints_3d=baseex['joints_3d'].to(device=device),
        joints_3d_visible=baseex['joints_3d_visible'].to(device=device),
        img_metas=None,
        return_loss=False)
codebook_base = res['codebook_indices']

codeword_replaces = []

for tokeni in range(token_num):

    pjoints = {}
    atol = 1e-6
    nsamples = 100
    codeword_replace = np.arange(dict_size,dtype=int)

    for cw1 in range(dict_size):
        codebook_curr = codebook_base.clone()
        codebook_curr[0,tokeni] = cw1
        with torch.no_grad():
            p1 = model.keypoint_head.tokenizer.decode_codewords(encoding_indices=codebook_curr.to(device=device))[0].cpu()
        ismatch = False
        for (cw0,p0) in pjoints.items():
            if torch.allclose(p0,p1,atol=atol):
                ismatch = True
                #print(f'codeword {cw0} and {cw1} are equivalent for base code')
                for samplei in range(nsamples):
                    codebook_curr0 = np.random.choice(dict_size,token_num)
                    codebook_curr0[tokeni] = cw0
                    codebook_curr0 = torch.tensor(codebook_curr0[None],device=device)
                    codebook_curr1 = codebook_curr0.clone()
                    codebook_curr1[0,tokeni] = cw1

                    with torch.no_grad():
                        p_joints0 = model.keypoint_head.tokenizer.decode_codewords(encoding_indices=codebook_curr0)[0]
                        p_joints1 = model.keypoint_head.tokenizer.decode_codewords(encoding_indices=codebook_curr1)[0]
                    if not torch.allclose(p_joints0,p_joints1,atol=atol):
                        ismatch = False
                        #print(f'codeword {cw0} and {cw1} are not equivalent for sample {samplei}')
                        break
            if ismatch:
                codeword_replace[cw1] = cw0
                #print(f'codeword {cw0} and {cw1} are equivalent for all samples, replace {cw1} -> {cw0}')
                break
        if not ismatch:
            pjoints[cw1] = p1
            #print(f'codeword {cw1} is unique, adding to pjoints')
    
    print(f'token {tokeni}, total number of unique codewords: {len(pjoints)}')
    codeword_replaces.append(codeword_replace)
    
# it is weird that the same codewords are used for each token, but i think my code is correct...

# %%
# how many unique codewords are actually used
all_unique_codewords = {}

for token_num_curr in token_nums:
    for dict_size_curr in dict_sizes:

        configfilecurr = configfiles[(token_num_curr, dict_size_curr)]
        cfgcurr = Config.fromfile(configfilecurr)
        
        # load in the model
        checkpointcurr,epochcurr = utils.get_last_checkpoint(configfilecurr,outdir)

        # load model
        modelcurr = build_posenet(cfgcurr.model)
        modelcurr = modelcurr.to(device)
        if checkpointcurr is not None:
            # load model checkpoint
            load_checkpoint(modelcurr, checkpointcurr, map_location=device)

        # what can the encoder produce
        unique_codewords = utils.compute_unique_codewords(modelcurr,train_dataloader)
        all_unique_codewords[(token_num_curr,dict_size_curr)] = unique_codewords
        for tokeni in range(token_num_curr):
            print(f'ntokens = {token_num_curr}, dictsize = {dict_size_curr}, token {tokeni}, total number of unique codewords: {len(unique_codewords[tokeni])}')


# %%
print('N. tokens\tDict. size\tN. codewords')
for (token_num_curr,dict_size_curr) in all_unique_codewords.keys():
    nuniquecodewords = len(list(set().union(*all_unique_codewords[(token_num_curr,dict_size_curr)])))
    print(f'{token_num_curr}\t{dict_size_curr}\t{nuniquecodewords}')


# %%
# get dict size for pre-trained coco data
cococonfigfile = 'configs/pct_base_woimgguide_tokenizer.py'
cococheckpoint = 'weights/tokenizer/swin_base_woimgguide.pth'
cococfg = Config.fromfile(cococonfigfile)
cocodataset = build_dataset(cococfg.data.train)
cocodataloader = build_dataloader(cocodataset,
                                    samples_per_gpu=batch_size,
                                    workers_per_gpu=1,
                                    num_gpus=1,
                                    dist=False,
                                    shuffle=False,
                                    seed=None,
                                    drop_last=False,
                                    pin_memory=True)
modelcurr = build_posenet(cococfg.model)
modelcurr = modelcurr.to(device)
load_checkpoint(modelcurr, cococheckpoint, map_location=device)
unique_codewords_coco = utils.compute_unique_codewords(modelcurr,cocodataloader)


# %%
# nbasepercat = 12
# behaviors = testann['info']['behaviors']
# print(behaviors)
# tokensplot = 0

# behkey,behname = next(iter(behaviors.items()))
# exidx = np.array([i for i,ex in enumerate(testann['annotations']) if ex['behavior'] == int(behkey)])
# print(f'n {behname} test examples: {len(exidx)}')
# exis = np.random.choice(exidx,nbasepercat,replace=False)
# p_joints = []
# codebook_order_nz = []
# for exi in exis:
#     behex = test_dataset[exi]
#     with torch.no_grad():
#         res = model(img=None,
#             joints_3d=torch.tensor(behex['joints_3d'][None]).to(device=device),
#             joints_3d_visible=torch.tensor(behex['joints_3d_visible'][None]).to(device=device),
#             img_metas=None,
#             return_loss=False)
#     codebook_beh_base = res['codebook_indices']
#     p_joints_curr,codebook_order_curr,dpath_curr,dtotal_curr = visualize_token(tokensplot,codebook_beh_base,dict_size,model,device)
#     if exi == 0:
#         codebook_order = codebook_order_curr
#     p_joints.append(p_joints_curr)
    
# ani = show_token_animation(p_joints,[codebook_order,]*nbasepercat,cfg,flyskeledges)

# # write animation to video 
# outfigdir = os.path.join(outdir,'figs')
# os.makedirs(outfigdir,exist_ok=True)
# outvidfile = os.path.join(outfigdir,f'tokenvis_base{behname}{exi}_ntokens{token_num}_dictsize{dict_size}.mp4')
# print(f'Saving example for behavior {behname} example {exi} ({exii}/{nbasepercat}) to {outvidfile}')
# # use pillow writer to write mp4
# ani.save(outvidfile, writer='ffmpeg', fps=10)

