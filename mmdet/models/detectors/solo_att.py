from .single_stage_ins import SingleStageInsDetector
from ..registry import DETECTORS
import torch
from mmdet.core import bbox2result
import numpy as np

def merge_outputs(detections, num_classes):
#     print(detections)
    results = {}
    max_per_image = 100
    for j in range(1, num_classes + 1):
        results[j] = detections[j-1]
#         if len(self.scales) > 1 or self.opt.nms:
        results[j] = soft_nms(results[j], Nt=0.5, method=2, threshold=0.01)
#         print(results)
    scores = np.hstack([results[j][:, 4] for j in range(1, num_classes + 1)])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
#     print("after merge out\n", results)
    return results2coco_boxes(results, num_classes)

def results2coco_boxes(results, num_classes): 
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class
    Returns:
        list(ndarray): bbox results of each class
    """
    bboxes = [0 for i in range(num_classes)]
    for j in range(1, num_classes + 1):
        if len(results[j]) == 0:
            bboxes[j - 1] = np.zeros((0, 5), dtype=np.float32)
            continue
        bboxes[j - 1] = results[j]
#     print(bboxes) # xyxy
    return bboxes


def soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.01, method=0):
    N = boxes.shape[0]
#     cdef float iw, ih, box_area
#     cdef float ua
#     cdef int 
    pos = 0
#     cdef float 
    maxscore = 0
#     cdef int 
    maxpos = 0
#     cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
                                
                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    boxes = boxes[keep]
    return boxes


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
            results = merge_outputs(bbox_results[0], 80)
            return results
            #return bbox_results[0]

    
    
