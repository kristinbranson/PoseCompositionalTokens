# %%
# %load_ext autoreload
# %autoreload 2

import os
import matplotlib.pyplot as plt
import numpy as np
import models
from mmcv import Config
import copy

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(models.__file__))))
plt.rcParams['savefig.directory'] = os.getcwd()

inconfigfile = 'configs/fly_pct_base_noaug_tokenizer.py'
cfg = Config.fromfile(inconfigfile)


# %%
token_nums = [1,2,4,8,16,32]
token_class_nums = [256,512,1024,2048]
cfgfiles = {}

for token_num in token_nums:
    for token_class_num in token_class_nums:
        outconfigfile = f'configs/fly_pct_base_ntokens{token_num}_dictsize{token_class_num}_tokenizer.py'
        cfgfiles[(token_num, token_class_num)] = outconfigfile
        if os.path.exists(outconfigfile):
            continue
        newcfg = copy.deepcopy(cfg)
        newcfg.model['keypoint_head']['tokenizer']['codebook']['token_num'] = token_num
        newcfg.model['keypoint_head']['tokenizer']['codebook']['token_class_num'] = token_class_num
        newcfg.dump(outconfigfile)


# %%
def configfile_to_outdir(outconfigfile):
    outdir = os.path.join(os.getcwd(),'work_dirs',os.path.basename(outconfigfile).replace('.py',''))
    return outdir


# %%
import datetime
gpuqueue = 'gpu_l4'
isrunning = {}
# get current timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')


# %%
for i,(k,outconfigfile) in enumerate(cfgfiles.items()):
    outdir = configfile_to_outdir(outconfigfile)
    if not os.path.exists(outdir):
        print(f'mkdir {outdir}')
        os.makedirs(outdir)
    outfile = os.path.join(outdir,f'log_{timestamp}.txt')
    jobname = f'ntokens{k[0]}_dictsize{k[1]}_tokenizer'
    nodecmd = "source ~/.bashrc; "
    nodecmd += f"cd {os.getcwd()}; "
    nodecmd += f"conda activate PCT; "
    
    nodecmd += f'python -m torch.distributed.launch --nproc_per_node=1 --master_port={29600+i} ./tools/train.py {outconfigfile} --launcher pytorch'
    #nodecmd += f"./tools/dist_train.sh {outconfigfile} 1 "
    submitcmd = f'bsub -n 1 -gpu "num=1" -q {gpuqueue} -J {jobname} -o "{outfile}" -L /bin/bash "{nodecmd}"'
    cmd = f"ssh login1 '{submitcmd}'"
    print(cmd)
    # Run and capture output
    if (k not in isrunning) or (not isrunning[k]):
        os.system(cmd)
        print('submitted')
        isrunning[k] = True

# %%
train_status = {}

for i,(k,outconfigfile) in enumerate(cfgfiles.items()):
    outdir = configfile_to_outdir(outconfigfile)
    # find files in outdir named epoch_*.pth and parse the epoch number
    files = os.listdir(outdir)
    epoch_nums = []
    for file in files:
        if 'epoch_' in file:
            epoch_num = int(file.split('_')[-1].split('.')[0])
            epoch_nums.append(epoch_num)
    train_status[k] = {}
    if len(epoch_nums) == 0:
        train_status[k]['epoch'] = np.nan
    else:
        last_epoch = max(epoch_nums)
        train_status[k]['epoch'] = last_epoch
    # find all log files named log_*.txt and parse the timestamp
    files = os.listdir(outdir)
    timestamps = []
    for file in files:
        if 'log_' in file:
            timestamp = file.split('_')[1].split('.')[0]
            timestamps.append(timestamp)
    if len(timestamps) == 0:
        train_status[k]['timestamp'] = ''
    else:
        last_timestamp = max(timestamps)
        train_status[k]['timestamp'] = last_timestamp
    print(f'{k}: {train_status[k]}')
    


# %%
def load_train_res(trainresjsonfile):

    #trainresjsonfile = 'work_dirs/fly_pct_base_noaug_tokenizer/20241216_181551.log.json'
    #trainresjsonfile = 'work_dirs/fly_pct_base_noaug_tokenizer/20241216_094019.log.json'
    with open(trainresjsonfile,'r') as f:
        inres = [json.loads(line) for line in f]
    trainres = {}
    for r in inres:
        if 'mode' not in r:
            continue
        ks = r.keys()
        if r['mode'] not in trainres:
            trainres[r['mode']] = {k:[] for k in ks if k != 'mode'}
        for k in ks:
            if k != 'mode':
                trainres[r['mode']][k].append(r[k])
    for mode in trainres.keys():
        for k,v in trainres[mode].items():
            trainres[mode][k] = np.array(v)
            
    return trainres


# %%
trainres = {}
for k,configfile in cfgfiles.items():
    # find all files named <timestamp>.log.json and parse out timestamp
    outdir = configfile_to_outdir(configfile)
    files = os.listdir(outdir)
    last_timestamp = None
    for file in files:
        if '.log.json' in file:
            timestamp = file.split('.')[0]
            if last_timestamp is None or timestamp > last_timestamp:
                last_timestamp = timestamp
    
    assert last_timestamp is not None
    
    trainresjsonfile = os.path.join(outdir,f'{last_timestamp}.log.json')
    trainres[k] = load_train_res(trainresjsonfile)

# %%
ntokens_try = len(token_nums)
ndictsizes_try = len(token_class_nums)
collected_res = {}
train_keys = ['joint_loss','e_latent_loss','loss']
val_keys = ['AP','AP .5','AP .75','AR','AR .5','AR .75']
collected_res['train'] = {}
for k in train_keys:
    collected_res['train'][k] = np.zeros((ntokens_try,ndictsizes_try))
collected_res['val'] = {}
for k in val_keys:
    collected_res['val'][k] = np.zeros((ntokens_try,ndictsizes_try))

for i,((ntokens,dictsize),v) in enumerate(trainres.items()):
    tokeni = token_nums.index(ntokens)
    dicti = token_class_nums.index(dictsize)
    for k in train_keys:
        collected_res['train'][k][tokeni,dicti] = v['train'][k][-1]
    for k in val_keys:
        collected_res['val'][k][tokeni,dicti] = v['val'][k][-1]

# %%
import prettytable
tb = prettytable.PrettyTable()
# make a table that has a column for each dict size and row for each n tokens, and shows the train joint loss in each entry
tb.field_names = ['n tokens'] + [f'{d}' for d in token_class_nums]
for i,n in enumerate(token_nums):
    row = [n] + list(collected_res['train']['joint_loss'][i,:])
    tb.add_row(row)
print('train joint loss')
print(tb)

tb = prettytable.PrettyTable()
tb.field_names = ['n tokens'] + [f'{d}' for d in token_class_nums]
for i,n in enumerate(token_nums):
    row = [n] + list(collected_res['train']['loss'][i,:])
    tb.add_row(row)
print('train loss')
print(tb)

tb = prettytable.PrettyTable()
tb.field_names = ['n tokens'] + [f'{d}' for d in token_class_nums]
for i,n in enumerate(token_nums):
    row = [n] + list(collected_res['val']['AP'][i,:])
    tb.add_row(row)
print('val AP')
print(tb)

tb = prettytable.PrettyTable()
tb.field_names = ['n tokens'] + [f'{d}' for d in token_class_nums]
for i,n in enumerate(token_nums):
    row = [n] + list(collected_res['val']['AR'][i,:])
    tb.add_row(row)
print('val AR')
print(tb)

fig,axs = plt.subplots(2,3,figsize=(15,10))

ax = axs[:,0]
for tokeni in range(ntokens_try):
    ntokens = token_nums[tokeni]
    ax[0].plot(token_class_nums,collected_res['train']['joint_loss'][tokeni,:],'o-',label=f'{ntokens}')
ax[0].legend(title='n tokens')
ax[0].set_xlabel('dict size')
ax[0].set_ylabel('train joint loss')
for dicti in range(ndictsizes_try):
    dictsize = token_class_nums[dicti]
    ax[1].plot(token_nums,collected_res['train']['joint_loss'][:,dicti],'o-',label=f'{dictsize}')
ax[1].legend(title='dict size')
ax[1].set_xlabel('n tokens')
ax[1].set_ylabel('train joint loss')

ax = axs[:,1]
for tokeni in range(ntokens_try):
    ntokens = token_nums[tokeni]
    ax[0].plot(token_class_nums,collected_res['val']['AP'][tokeni,:],'o-',label=f'{ntokens}')
ax[0].legend(title='n tokens')
ax[0].set_xlabel('dict size')
ax[0].set_ylabel('val AP')
for dicti in range(ndictsizes_try):
    dictsize = token_class_nums[dicti]
    ax[1].plot(token_nums,collected_res['val']['AP'][:,dicti],'o-',label=f'{dictsize}')
ax[1].legend(title='dict size')
ax[1].set_xlabel('n tokens')
ax[1].set_ylabel('val AP')

ax = axs[:,2]
for tokeni in range(ntokens_try):
    ntokens = token_nums[tokeni]
    ax[0].plot(token_class_nums,collected_res['val']['AR'][tokeni,:],'o-',label=f'{ntokens}')
ax[0].legend(title='n tokens')
ax[0].set_xlabel('dict size')
ax[0].set_ylabel('val AR')
for dicti in range(ndictsizes_try):
    dictsize = token_class_nums[dicti]
    ax[1].plot(token_nums,collected_res['val']['AR'][:,dicti],'o-',label=f'{dictsize}')
ax[1].legend(title='dict size')
ax[1].set_xlabel('n tokens')
ax[1].set_ylabel('val AR')
