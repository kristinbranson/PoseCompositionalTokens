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
#     display_name: PCT
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import json
import models
import os
from collections import OrderedDict
import numpy as np
from scipy.spatial import ConvexHull

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(models.__file__))))


# %%
# add area to annotations

indatadir = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024'
outdatadir = 'data/fly_bubble_data_20241024'
if not os.path.exists(outdatadir):
    os.makedirs(outdatadir)
    
for filetype in ['train','test']:
    inannfile = os.path.join(indatadir,filetype+'_annotations.json')
    outannfile = os.path.join(outdatadir,filetype+'_annotations.json')
    if os.path.exists(outannfile):
        print(f'{outannfile} already exists, skipping')
        continue
    inann = json.load(open(inannfile))

    for anncurr in inann['annotations']:
        x = anncurr['keypoints'][0::3]
        y = anncurr['keypoints'][1::3]
        p = np.array([x,y]).T
        hull = ConvexHull(p)
        area = hull.volume
        anncurr['area'] = area

    # output
    with open(outannfile, 'w') as f:
        json.dump(inann, f)
        
# make a symbolic link to image directories
for filetype in ['train','test']:
    if not os.path.exists(f'{outdatadir}/{filetype}'):
        os.system(f'ln -s {indatadir}/{filetype} {outdatadir}/{filetype}')

# %%
# read train annotations

inannfile = os.path.join(outdatadir,'train_annotations.json')
ann = json.load(open(inannfile))
print(ann['info'].keys())
kpnames = ann['categories'][0]['keypoints']
skeledges = ann['categories'][0]['skeleton']
print(f'kpnames: {kpnames}')
print(f'skeledges: {skeledges}')

# %%
for i,(kp1,kp2) in enumerate(skeledges):
    print(f'{i}: {kpnames[kp1-1]} -> {kpnames[kp2-1]}')

# %%
# create dataset_info dict

dataset_info = {}
# get parent directory name
dataset_info['dataset_name'] = os.path.basename(os.path.dirname(annfile))
dataset_info['paper_info'] = {
    'authors': 'Alice A. Robie and Adam L. Taylor and Catherine E. Schretter and Mayank Kabra and Kristin Branson',
    'title': 'The Fly Disco: Hardware and software for optogenetics and fine-grained fly behavior analysis',
    'year': '2024',
    'homepage': 'https://research.janelia.org/bransonlab/multifly',
}
print(dataset_info['dataset_name'])
print(dataset_info['paper_info'])

# fix typo
kpbad = 'right_mid_fitib'
if kpbad in kpnames:
    ibad = kpnames.index('right_mid_fitib')
    kpnames[ibad] = 'right_mid_fetib'

# choose colors for kpts
cm = plt.get_cmap('tab10')
colors = cm.colors
# fig,ax = plt.subplots()
# for i,c in enumerate(colors):
#     ax.plot([i,i],[i,i],'o',ms=20,color=c,label=f'{i}')
#     print(f'{i}: {c}')
# ax.axis('auto')
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

for k in kpnames:
    print(k)

alpha = .75
def darken(c):
    return [x*alpha for x in c]
def lighten(c):
    return [x*alpha+(1-alpha) for x in c]

kp2color = {'ant_head': colors[0],
            'left_eye': darken(colors[0]),
            'right_eye': lighten(colors[0]),
            'left_thorax': darken(colors[1]),
            'right_thorax': lighten(colors[1]),
            'pos_notum': colors[1],
            'pos_abdomen': colors[1],
            'left_front_tar': darken(colors[2]),
            'right_front_tar': lighten(colors[2]),
            'left_mid_fe': darken(colors[3]),
            'left_mid_fetib': darken(colors[3]),
            'left_mid_tar': darken(colors[3]),
            'right_mid_fe': lighten(colors[3]),
            'right_mid_fetib': lighten(colors[3]),
            'right_mid_tar': lighten(colors[3]),
            'left_back_tar': darken(colors[4]),
            'right_back_tar': lighten(colors[4]),
            'left_outer_wing': darken(darken(colors[9])),
            'left_mid_wing': darken(colors[9]),
            'right_outer_wing': lighten(lighten(colors[9])),
            'right_mid_wing': lighten(colors[9]),
}
fig,ax = plt.subplots()
tmpann = ann['annotations'][0]['keypoints']
nkpts = len(kpnames)
for (k,v) in kp2color.items():
    i = kpnames.index(k)
    ax.plot(tmpann[i*3],tmpann[i*3+1],'o',ms=10,color=v,label=k)
ax.axis('auto')
ax.axis('equal')
ax.invert_yaxis()
for k in kpnames:
    assert k in kp2color.keys()
    
dataset_info['keypoint_info'] = {}
for i,k in enumerate(kpnames):
    # if k starts with 'right':
    if k.startswith('right'):
        swapk = k.replace('right','left')
        assert swapk in kpnames
    elif k.startswith('left'):
        swapk = k.replace('left','right')
        assert swapk in kpnames
    else:
        swapk = k
        
    kpinfo = {
        'name':k,
        'id': i,
        'color': list(kp2color[k]),
        'swap': swapk,
    }
    dataset_info['keypoint_info'][i] = kpinfo
for k,v in dataset_info['keypoint_info'].items():
    print(f'{k}: {v}')
    
    
dataset_info['skeleton_info'] = {}
for i,(a,b) in enumerate(skeledges):
    kp1 = kpnames[a-1]
    kp2 = kpnames[b-1]
    c1 = kp2color[kp1]
    c2 = kp2color[kp2]
    c = [x for x in c2]
    edgeinfo = {
        'link': (kp1,kp2),
        'id': i,
        'color': c,
        }
    dataset_info['skeleton_info'][i] = edgeinfo
    
dataset_info['joint_weights'] = [1.,]*nkpts

# sigma = sqrt(E[dist(true,pred)^2/convhull(true)])
# computed in AnimalPoseForecasting/notebooks/choose_discretize_bins.py
dataset_info['sigmas'] = \
    [0.00900317, 0.01410282, 0.01521308, 0.01259913, 0.01176287, 0.02467385,
    0.01929638, 0.01751823, 0.02162774, 0.01829655, 0.02305875, 0.04583739,
    0.04365924, 0.06171027, 0.06218687, 0.04949219, 0.04573409, 0.01876062,
    0.02085872, 0.02960416, 0.0278293 ]
print(dataset_info)

# %%
# output

import pprint
from typing import Dict, Any
outconfigfile = 'configs/fly_bubble_data_20241024.py'

def create_config_file(dataset_info: Dict[str, Any], outconfigfile: str) -> None:
    """
    Creates a config.py file containing the provided dictionary.
    
    Args:
        dataset_info (Dict[str, Any]): Dictionary containing configuration information
        
    The function writes a Python file that recreates the dictionary when executed.
    """
    with open(outconfigfile, 'w') as f:
        f.write("# Auto-generated configuration file\n\n")
        
        # Write the dictionary definition
        f.write("dataset_info = ")
        
        # Use pprint to format the dictionary nicely
        formatted_dict = pprint.pformat(dataset_info, indent=2, width=100, sort_dicts=False)
        
        # Write the formatted dictionary
        f.write(formatted_dict)
        
        # Add newline at end of file
        f.write("\n")
        
create_config_file(dataset_info,outconfigfile)    

# %%
# print flip indices
dataset_info['keypoint_info']
FLIPINFO = [None,]*nkpts
for k,v in dataset_info['keypoint_info'].items():
    print(f"{k}: {v['name']} -> {v['swap']} = {kpnames.index(v['swap'])}")
    FLIPINFO[k] = kpnames.index(v['swap'])
print('FLIPINFO:')
print(FLIPINFO)

# %%
# output bbox file
outbboxfile = 'data/fly_bubble_data_20241024/test_detections.json'
annfile = f'data/fly_bubble_data_20241024/test_annotations.json'

ann = json.load(open(annfile))

outbboxes = []
for anncurr in ann['annotations']:
    bbox = {
        'bbox': anncurr['bbox'],
        'category_id': 1,
        'image_id': anncurr['image_id'],
        'score': 1.0,
    }
    outbboxes.append(bbox)

with open(outbboxfile,'w') as f:
    json.dump(outbboxes,f)

#{'bbox': [249.8199079291458, 175.21093805640606, 74.00419360691592, 55.626325589288854], 'category_id': 1, 'image_id': 532481, 'score': 0.9992738366127014}

# %%
# for debugging, make a version of ann with just 17 keypoints
newnkpts = 17

for name in ['train','test']:

    annfile = f'data/fly_bubble_data_20241024/{name}_annotations.json'
    ann = json.load(open(annfile))

    ann['info']['description'] = 'Fly Bubble training data, 17 keypoints'
    ann['info']['date_created'] = '2024-12-12'
    for i in range(len(ann['annotations'])):
        if len(ann['annotations'][i]['keypoints']) > newnkpts*3:
            ann['annotations'][i]['keypoints'] = ann['annotations'][i]['keypoints'][:newnkpts*3]
            ann['annotations'][i]['num_keypoints'] = newnkpts
        assert len(ann['annotations'][i]['keypoints']) == newnkpts*3

    outannfile = f'data/fly_bubble_data_20241024/{name}_annotations_17kp.json'
    with open(outannfile, 'w') as f:
        json.dump(ann,f)

# %%
# make a version of the config file with just 17 keypoints
print(dataset_info.keys()) 
dataset_info['dataset_name'] = 'fly_bubble_data_20241024_17kp'
dataset_info['keypoint_info'] = {k:dataset_info['keypoint_info'][k] for k in range(newnkpts)}

newkpnames = [kp['name'] for kp in dataset_info['keypoint_info'].values()]
newskelinfo = {}
for k,v in dataset_info['skeleton_info'].items():
    if v['link'][0] in newkpnames and v['link'][1] in newkpnames:
        newskelinfo[len(newskelinfo)] = v
dataset_info['skeleton_info'] = newskelinfo

dataset_info['joint_weights'] = [1.,]*newnkpts
dataset_info['sigmas'] = dataset_info['sigmas'][:newnkpts]

outconfigfile = 'configs/fly_bubble_data_20241024_17kp.py'
create_config_file(dataset_info,outconfigfile)   

# %%
# print flip indices
dataset_info['keypoint_info']
FLIPINFO = [None,]*newnkpts
for k,v in dataset_info['keypoint_info'].items():
    print(f"{k}: {v['name']} -> {v['swap']} = {newkpnames.index(v['swap'])}")
    FLIPINFO[k] = newkpnames.index(v['swap'])
print('FLIPINFO (17kp):')
print(FLIPINFO)

# %%
import torch
from models import build_posenet
from mmcv import Config
from mmpose.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint

# read the configfile
configfile = 'configs/dummy_fly_pct_base_woimgguide_tokenizer.py'
cfg = Config.fromfile(configfile)

# create the dataset
dataset = build_dataset(cfg.data.test)

# %%
import numpy as np

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
ex = dataset[0]
fig,ax = plt.subplots()
ax.imshow(ex['img'].permute(1,2,0))
ex['joints_3d'].shape
plot_skeleton(ex['joints_3d'],ax=ax)

# %%
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

checkpoint = 'weights/tokenizer/swin_base_woimgguide.pth'

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
with torch.no_grad():
    res = model(img=None,
                joints_3d=exbatch['joints_3d'].to(device=device),
                joints_3d_visible=exbatch['joints_3d_visible'].to(device=device),
                img_metas=None,
                return_loss=False)

