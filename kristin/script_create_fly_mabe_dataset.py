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
import matplotlib.pyplot as plt
import json
import models
import os
from collections import OrderedDict
import numpy as np
from scipy.spatial import ConvexHull
import re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(models.__file__))))
datadir = 'data/fly_mabe2022_v2'



# %%
# read train annotations

inannfile = os.path.join(datadir,'testtrain_v2_sample10000.json')
ann = json.load(open(inannfile))
print(ann['info'].keys())
kpnames = ann['categories'][0]['keypoints']
skeledges = ann['categories'][0]['skeleton']
print(f'kpnames: {kpnames}')
print(f'skeledges: {skeledges}')

# %%
for i,(kp1,kp2) in enumerate(skeledges):
    print(f'{i}: {kpnames[kp1-1]} -> {kpnames[kp2-1]}')
for kp in kpnames:
    print(f'{kp}')

# %%
# create dataset_info dict

dataset_info = {}
# get parent directory name
dataset_info['dataset_name'] = os.path.basename(os.path.dirname(inannfile))
dataset_info['paper_info'] = {
    'authors': 'Jennifer J. Sun, Markus Marks, Andrew Ulmer, Dipam Chakraborty, Brian Geuther, Edward Hayes, Heng Jia, Vivek Kumar, Sebastian Oleszko, Zachary Partridge, Milan Peelman, Alice Robie, Catherine Schretter, Keith Sheppard, Chao Sun, Param Uttarwar, Julian Wagner, Erik Werner, Joseph Parker, Pietro Perona, Yisong Yue, Kristin Branson, Ann Kennedy',
    'title': 'MABe22: A Multi-Species Multi-Task Benchmark for Learned Representations of Behavior',
    'year': '2023',
    'homepage': 'https://sites.google.com/view/computational-behavior/our-datasets/mabe2022-dataset',
}
print(dataset_info['dataset_name'])
print(dataset_info['paper_info'])

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

kp2color = {'antennae_midpoint': colors[0],
            'left_eye': darken(colors[0]),
            'right_eye': lighten(colors[0]),
            'left_front_thorax': darken(colors[1]),
            'right_front_thorax': lighten(colors[1]),
            'base_thorax': colors[1],
            'tip_abdomen': colors[1],
            'left_front_leg_tip': darken(colors[2]),
            'right_front_leg_tip': lighten(colors[2]),
            'left_middle_femur_base': darken(colors[3]),
            'left_middle_femur_tibia_joint': darken(colors[3]),
            'left_middle_leg_tip': darken(colors[3]),
            'right_middle_femur_base': lighten(colors[3]),
            'right_middle_femur_tibia_joint': lighten(colors[3]),
            'right_middle_leg_tip': lighten(colors[3]),
            'left_back_leg_tip': darken(colors[4]),
            'right_back_leg_tip': lighten(colors[4]),
            'left_outer_wing': darken(darken(colors[9])),
            'wing_left': darken(colors[9]),
            'right_outer_wing': lighten(lighten(colors[9])),
            'wing_right': lighten(colors[9]),
}
fig,ax = plt.subplots()
tmpann = ann['annotations'][0]['keypoints']
nkpts = len(kpnames)
for (k,v) in kp2color.items():
    i = kpnames.index(k)
    ax.plot(tmpann[i*3],tmpann[i*3+1],'o',ms=10,color=v,label=k)
for i,(a,b) in enumerate(skeledges):
    kp1 = kpnames[a-1]
    kp2 = kpnames[b-1]
    c2 = kp2color[kp2]
    c = [x for x in c2]
    ax.plot([tmpann[a*3],tmpann[b*3]],
            [tmpann[a*3+1],tmpann[b*3+1]],
            color=c)
ax.axis('auto')
ax.axis('equal')
ax.invert_yaxis()
for k in kpnames:
    assert k in kp2color.keys()
    
dataset_info['keypoint_info'] = {}
for i,k in enumerate(kpnames):
    # if k contains 'right'
    if re.match(r'.*right.*',k):
        swapk = k.replace('right','left')
        assert swapk in kpnames
    elif re.match(r'.*left.*',k):
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
outconfigfile = 'configs/fly_mabe_data_v2_20250106.py'

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
