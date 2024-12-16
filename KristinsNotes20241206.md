# Notes on trying Pose Compositional Tokens

Followed README.md:
* Created conda env PCT - required modifications to requirements.txt
* Downloaded data - COCO already downloaded, made symlink, downloaded HRNet results here
/groups/branson/bransonlab/datasets/coco
* Trained tokenizer:
```
./tools/dist_train.sh configs/pct_base_tokenizer.py 1
```
This is with image guidance, probably want without.
* Could run demo on my images
```
https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth
```

## training tokenizer

dataset is an mmpose dataset, which looks a fair amount like a pytorch dataset
dataset[0] -> dict
```
dataset[0].keys()
dict_keys(['img', 'joints_3d', 'joints_3d_visible', 'img_metas'])
dataset[0]['img'].shape
torch.Size([3, 256, 256])
dataset[0]['joints_3d'].shape
(17, 3)
dataset[0]['joints_3d'][0,:]
array([86.5159 , 33.23323,  0.     ], dtype=float32)
dataset[0]['joints_3d_visible'].shape
(17, 3)
dataset[0]['joints_3d_visible'][0,:]
array([1., 1., 0.], dtype=float32)
dataset[0]['img_metas']
```

There is a different config for train, val, and test data, which differ in the data source and processing pipeline. train data has a lot more processing, which primarily looks like data augmentation?

Made demo script that inputs a test coco example and just runs the tokenizer (as far as I can tell): `demos/demo_tokenizer.py`. Here is an output image:
![](images/demo_tokenizer_fig.png)

Modified pct_detector.py PCT.forward_test so that it could function without an img.

## Created dataset from fly bubble data

There are three types of files:
* **PCT config file**: specifies network architecture, data locations, data data processing. Example: `configs/fly_pct_base_woimgguide_tokenizer.py`. This points to the mmpose dataset config file with the `_base_` key and the locations of the coco-formatted train, val, and test data with the `data` field. 
* **mmpose dataset config file**: specifies metadata about the dataset. This is standalone, doesn't reference the dataset annotation files.
* **coco-format annotation json files**: COCO-formatted training, val, and test data. Specifies the annotations and the locations of the images. 

Wrote `kristin/script_create_fly_bubble_dataset.py` to make an mmpose-compatible coco-style config.

I made a symlink to the fly bubble data in data:
`data/fly_bubble_data_20241024 -> /groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024/`

I read in data from the train_annotations.ann file to figure out what `dataset_info` should be, and save this to the config file: `configs/fly_bubble_data_20241024.py`. `sigmas` were computed in `AnimalPoseForecasting/notebooks/choose_discretize_bins.py`

I made versions of the train and test annotations that only have 17 keypoints, since this is what the coco dataset has. I put these in:
* `data/fly_bubble_data_20241024/train_annotations_17kp.json`
* `data/fly_bubble_data_20241024/test_annotations_17kp.json`
(note that since the data dir is a sym link, they are also in `/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024/`).

The `dataset_info` config file with just 17 keypoints is `configs/fly_bubble_data_20241024_17kp.py`.

I copied `configs/pct_base_woimgguide_tokenizer.py` to make `configs/dummy_fly_pct_base_woimgguide_tokenizer.py`. This is set up for 17 keypoints. Main changes I made:
* Pointed to fly_bubble_data_20241024 dataset info config file
* Pointed to the fly_bubble_data_20241024 data locations
* Modified `dataset_name`.

I made a second version
`configs/dummy_fly_pct_base_woimgguide_tokenizer_v2.py`
that doesn't do any image loading or processing. I'd like to make the dataset not have to read in an image, but maybe that is too much work...

In demo_tokenizer, I also tried running the pre-trained human tokenizer on fly poses:
![](images/demo_tokenizer_fly_fig.png)