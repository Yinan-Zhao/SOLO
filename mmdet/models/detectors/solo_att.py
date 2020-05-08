from .single_stage_ins import SingleStageInsDetector
from ..registry import DETECTORS
import torch
from mmdet.core import bbox2result


@DETECTORS.register_module
class SOLOAtt(SingleStageInsDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SOLOAtt, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x = self.extract_feat(img)
        loss_inputs = (x, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    '''def simple_test(self, img, img_meta, rescale=False):
        with torch.no_grad():
            x = self.extract_feat(img)
            seg_inputs = (x, img_meta, self.test_cfg, rescale)
            seg_result = self.bbox_head.get_seg(*seg_inputs)
            return seg_result'''  

    def simple_test(self, img, img_meta, rescale=False):
        with torch.no_grad():
            x = self.extract_feat(img)
            seg_inputs = (x, img_meta, self.test_cfg, rescale)
            bbox_list = self.bbox_head.get_seg(*seg_inputs)
            bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            return bbox_results[0]
    
