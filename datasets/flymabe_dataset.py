import copy
# import warnings
# from abc import ABCMeta, abstractmethod

# import json_tricks as json
# import numpy as np
# from torch.utils.data import Dataset
from xtcocotools.coco import COCO

# from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
#                                                   keypoint_nme,
#                                                   keypoint_pck_accuracy)
# from mmpose.datasets import DatasetInfo
# from mmpose.datasets.pipelines import Compose

from mmpose.datasets.datasets.top_down import TopDownCocoDataset
from mmpose.datasets.builder import DATASETS
import pickle

import numpy as np
from collections import OrderedDict
from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_nme,
                                                  keypoint_pck_accuracy)


@DATASETS.register_module()
class FlyMABe2022Dataset(TopDownCocoDataset):
    """
    FlyMABE2022 dataset for trajectories from Fly MABE 2022 Challenge
    """
    
    def __init__(self,ann_file,img_prefix,data_cfg,pipeline,dataset_info=None,test_mode=False):
        super(TopDownCocoDataset, self).__init__(
            ann_file, # can be pkl or json
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode,
            coco_style=False)

        # copied from coco_style section of Kpt2dSviewRgbImgTopDownDataset.__init__
        # with the minor change that it can load from a pkl file 

        # if ann_file extension is pkl
        if ann_file.endswith('.pkl'):
            # load in the pkl file
            with open(ann_file, 'rb') as f:
                ann_data = pickle.load(f)
            self.coco = COCO(annotation_file=ann_file,ann_data=ann_data)
        else:
            self.coco = COCO(annotation_file=ann_file)

        if 'categories' in self.coco.dataset:
            cats = [
                cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())
            ]
            self.classes = ['__background__'] + cats
            self.num_classes = len(self.classes)
            self._class_to_ind = dict(
                zip(self.classes, range(self.num_classes)))
            self._class_to_coco_ind = dict(
                zip(cats, self.coco.getCatIds()))
            self._coco_ind_to_class_ind = dict(
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:])
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(
            self.coco.imgs)

        # copied from TopDownCocoDataset.__init__
        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')
        
    def _get_db(self):
        """Load dataset."""
        gt_db = super()._get_db()
        
        # add in mov, frm, id
        for item,ann in zip(gt_db,self.coco.dataset['annotations']):
            item['id'] = ann['id']
            item['mov'] = ann['mov']
            item['frm'] = ann['frm']
            
        return gt_db
        
    def evaluate(self, results, res_folder=None, metric='mAP', **kwargs):
        
        allowed_metrics = ['mAP','EPE']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
                
        if metric == 'mAP':
            return super().evaluate(results,
                                    res_folder=res_folder,
                                    metric=metric,
                                    **kwargs)
        else:
            
            # box_sizes = []
            # threshold_bbox = []
            # threshold_head_box = []

            ntotal = np.sum([result['preds'].shape[0] for result in results])
            nkpts = results[0]['preds'].shape[1]
            d = results[0]['preds'].shape[2]-1
            preds = np.zeros((ntotal, nkpts, d),dtype=results[0]['preds'].dtype)
            off = 0
            for result in results:
                preds[off:off + result['preds'].shape[0]] = result['preds'][...,:-1]
                off += result['preds'].shape[0]

            gts = np.zeros((ntotal, nkpts, d),dtype=self.db[0]['joints_3d'].dtype)
            masks = np.zeros((ntotal, nkpts),dtype=bool)
            off = 0
            for item in self.db:
                gts[off:off + item['joints_3d'].shape[0]] = item['joints_3d'][...,:-1]
                masks[off:off + item['joints_3d_visible'].shape[0]] = item['joints_3d_visible'][:,0]>0
                off += item['joints_3d'].shape[0]
                # if metric == 'PCK':
                #     bbox = np.array(item['bbox'])
                #     bbox_thr = np.max(bbox[2:])
                #     threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
                # elif metric == 'PCKh':
                #     head_box_thr = item['head_size']
                #     threshold_head_box.append(
                #         np.array([head_box_thr, head_box_thr]))
                # box_sizes.append(item.get('box_size', 1))

            # outputs = np.array(outputs)
            # gts = np.array(gts)
            # masks = np.array(masks)
            # threshold_bbox = np.array(threshold_bbox)
            # threshold_head_box = np.array(threshold_head_box)
            # box_sizes = np.array(box_sizes).reshape([-1, 1])

            # if metric == 'PCK':
            #     _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
            #                                     threshold_bbox)
            #     info_str.append(('PCK', pck))

            # if 'PCKh' in metrics:
            #     _, pckh, _ = keypoint_pck_accuracy(outputs, gts, masks, pckh_thr,
            #                                     threshold_head_box)
            #     info_str.append(('PCKh', pckh))

            # if 'AUC' in metrics:
            #     info_str.append(('AUC', keypoint_auc(outputs, gts, masks,
            #                                         auc_nor)))

            if 'EPE' == metric:
                name_value = OrderedDict([('EPE', keypoint_epe(preds, gts, masks))])

            # if 'NME' in metrics:
            #     normalize_factor = self._get_normalize_factor(
            #         gts=gts, box_sizes=box_sizes)
            #     info_str.append(
            #         ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

            return name_value