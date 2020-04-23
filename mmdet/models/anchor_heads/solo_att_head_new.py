import mmcv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.ops import DeformConv, roi_align
from mmdet.core import multi_apply, bbox2roi, matrix_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.utils import _single, _pair, _triple
from functools import partial
from six.moves import map
import pdb

INF = 1e8

from scipy import ndimage

def multi_apply_custom(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return list(map_results)

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d


class MaskAttModule(nn.Module):
    def __init__(self, mask_in, out_channels):
        super(MaskAttModule, self).__init__()
        self.mask_conv = nn.Conv2d(mask_in, out_channels, 3, padding=1)
        self.att_conv = nn.Conv2d(1, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def init_weights(self):
        normal_init(self.mask_conv, std=0.01)
        normal_init(self.att_conv, std=0.01)


    def forward(self, mask_feat, attention, ins_ind_count):
        mask_output_raw = self.mask_conv(mask_feat)
        att_output = self.att_conv(attention)
        mask_output = []
        for i in range(len(ins_ind_count)):
            mask_output += [mask_output_raw[None,i] for j in range(ins_ind_count[i])]
        output = torch.cat(mask_output, dim=0) + att_output
        return self.relu(output)

class InstanceHead(nn.Module):
    def __init__(self, mask_in, out_channels, num_conv):
        super(InstanceHead, self).__init__()
        self.num_conv = num_conv
        self.mask_att_module = MaskAttModule(mask_in, out_channels)
        for i in range(num_conv):
            setattr(self, 'conv_%d'%(i), nn.Conv2d(out_channels, out_channels, 3, padding=1))
            setattr(self, 'relu_%d'%(i), nn.ReLU())
        self.conv_final = nn.Conv2d(out_channels, 1, 3, padding=1)

    def init_weights(self):
        self.mask_att_module.init_weights()
        for i in range(self.num_conv):
            conv_layer = getattr(self, 'conv_%d'%(i))
            normal_init(conv_layer, std=0.01)
        bias_inst = bias_init_with_prob(0.01)    
        normal_init(self.conv_final, std=0.01, bias=bias_inst)


    def forward(self, mask_feat, attention, ins_ind_count):
        # mask_feat: N,C,H,W (C is the number of channels)
        # attention: N,Ins,H,W (Ins is the number of instances)
        x = self.mask_att_module(mask_feat, attention, ins_ind_count)
        for i in range(self.num_conv):
            conv_layer = getattr(self, 'conv_%d'%(i))
            relu_layer = getattr(self, 'relu_%d'%(i))
            x = conv_layer(x)
            x = relu_layer(x)
        x = self.conv_final(x)
        return x


@HEADS.register_module
class SOLOAttHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 inst_feat_channels=8,
                 inst_convs=1,
                 stacked_convs=4,
                 local_mask_size=16,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.4,
                 cate_down_pos=0,
                 with_deform=False,
                 loss_ins=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SOLOAttHead, self).__init__()
        self.num_classes = num_classes
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.inst_feat_channels = inst_feat_channels
        self.num_inst_convs = inst_convs
        self.stacked_convs = stacked_convs
        self.local_mask_size = local_mask_size
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()


    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.feature_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        self.size_convs = nn.ModuleList()
        self.offset_convs = nn.ModuleList()
        self.localmask_convs = nn.ModuleList()
        self.inst_convs = InstanceHead(self.seg_feat_channels, self.inst_feat_channels, self.num_inst_convs)
        
        # mask feature (feature_convs)    
        for i in range(4):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = ConvModule(
                    self.in_channels,
                    self.seg_feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=norm_cfg is None)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.feature_convs.append(convs_per_level)
                continue
            for j in range(i):
                if j == 0:
                    in_channel = self.in_channels
                    one_conv = ConvModule(
                        in_channel,
                        self.seg_feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=norm_cfg is None)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue
                one_conv = ConvModule(
                    self.seg_feat_channels,
                    self.seg_feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=norm_cfg is None)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)
            self.feature_convs.append(convs_per_level) 

        # cate_convs
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels 
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        self.cate_convs.append(ConvModule(self.seg_feat_channels, self.cate_out_channels, 
                                            3, padding=1, activation=None, bias=True))

        # size_convs
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels 
            self.size_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        self.size_convs.append(ConvModule(self.seg_feat_channels, 2, 3, padding=1, activation=None, bias=True))

        # offset_convs
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels 
            self.offset_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        self.offset_convs.append(ConvModule(self.seg_feat_channels, 2, 3, padding=1, activation=None, bias=True))

        # localmask_convs
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.seg_feat_channels 
            self.localmask_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        self.localmask_convs.append(ConvModule(self.seg_feat_channels, self.local_mask_size**2, 
                                                3, padding=1, activation=None, bias=True))
        
        self.solo_mask = ConvModule(
            self.seg_feat_channels, self.seg_feat_channels, 1, padding=0, norm_cfg=norm_cfg, bias=norm_cfg is None)

 
    def init_weights(self):
        #TODO: init for feat_conv
        for m in self.feature_convs:
            s=len(m)
            for i in range(s):
                if i%2 == 0:
                    normal_init(m[i].conv, std=0.01)

        bias_cate = bias_init_with_prob(0.01)
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)      
        normal_init(self.cate_convs[-1].conv, std=0.01, bias=bias_cate)

        for m in self.size_convs:
            normal_init(m.conv, std=0.01)  

        for m in self.offset_convs:
            normal_init(m.conv, std=0.01)  

        for m in self.localmask_convs:
            normal_init(m.conv, std=0.01)  

        normal_init(self.solo_mask, std=0.01)
        self.inst_convs.init_weights()

    def create_zeros_as(self, x, shape):
        device = x.device
        target_type = x.dtype
        y = torch.zeros(shape, dtype=target_type, device=device)
        return y

    def get_att_single(self, scale_idx, feature_pred, idx_raw, is_eval=False):
        device = feature_pred.device
        target_type = feature_pred.dtype
        N, c, h, w = feature_pred.shape
        num_grid = self.seg_num_grids[scale_idx]
        att_template = self.att_pyramid[scale_idx]
        h_att, w_att = att_template.shape
        h_att_half = (h_att-1)//2
        w_att_half = (w_att-1)//2

        attention = torch.zeros([1, 1, h, w], dtype=target_type, device=device)
        idx = idx_raw % (num_grid**2)
        idx_h = idx // num_grid
        idx_w = idx % num_grid
        center_h = int(idx_h*h/num_grid)
        center_w = int(idx_w*w/num_grid)

        h_min_raw = center_h - h_att_half
        h_max_raw = center_h + h_att_half + 1
        w_min_raw = center_w - w_att_half
        w_max_raw = center_w + w_att_half + 1

        if h_min_raw < 0:
            h_min = 0
            h_att_min = h_min - h_min_raw
        else:
            h_min = h_min_raw
            h_att_min = 0
        if h_max_raw > h:
            h_max = h
            h_att_max = h_att - (h_max_raw-h)
        else:
            h_max = h_max_raw
            h_att_max = h_att

        if w_min_raw < 0:
            w_min = 0
            w_att_min = w_min - w_min_raw
        else:
            w_min = w_min_raw
            w_att_min = 0
        if w_max_raw > w:
            w_max = w
            w_att_max = w_att - (w_max_raw-w)
        else:
            w_max = w_max_raw
            w_att_max = w_att

        attention[0,0,h_min:h_max,w_min:w_max] = att_template[h_att_min:h_att_max,w_att_min:w_att_max]

        return attention

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'), 
                feats[1], 
                feats[2], 
                feats[3], 
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))


    def forward_mask_feat(self, feats):
        feature_add_all_level = self.feature_convs[0](feats[0]) 
        for i in range(1,4):
            feature_add_all_level = feature_add_all_level + self.feature_convs[i](feats[i])
        feature_pred = self.solo_mask(feature_add_all_level) 
        return feature_pred


    def forward_single_inst(self, mask_feat, attention, ins_ind_count, idx, featmap_size=None, is_eval=False):
        if attention.shape[0]:
            inst_pred = self.inst_convs(mask_feat, attention, ins_ind_count)
            if is_eval:
                inst_pred = inst_pred.sigmoid()
            else:
                assert featmap_size is not None
                inst_pred = F.interpolate(inst_pred, size=(featmap_size[0]*2,featmap_size[1]*2), mode='bilinear')
        else:
            inst_pred = self.create_zeros_as(mask_feat, [0,1,mask_feat[-2],mask_feat[-1]])

        return inst_pred


    def forward_single_cat(self, x, idx, is_eval=False):
        # cate head
        cate_feat = x
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)

        # size head
        size_feat = x
        for i, size_layer in enumerate(self.size_convs):
            size_feat = size_layer(size_feat)

        # offset head
        offset_feat = x
        for i, offset_layer in enumerate(self.offset_convs):
            offset_feat = offset_layer(offset_feat)

        # localmask head
        localmask_feat = x
        for i, localmask_layer in enumerate(self.localmask_convs):
            localmask_feat = localmask_layer(localmask_feat)

        if is_eval:
            cate_feat = points_nms(cate_feat.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_feat, size_feat, offset_feat, localmask_feat

    def loss(self,
             feats,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        featmap_sizes_pred = [(featmap.size()[-2]*2, featmap.size()[-1]*2) for featmap in
                         new_feats]

        # ins_ind_label_list is a list with only one element
        # ins_ind_label_list[0][0]: 1600,  ins_ind_label_list[0][1]: 1296
        # cate_label_list[0][0]: 40x40,  cate_label_list[0][1]: 36x36
        # ins_label_list[0][0]: 1600x288x192,  ins_label_list[0][0]:1296x288x192
        ins_label_list, cate_label_list, ins_ind_label_list = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            featmap_sizes=featmap_sizes_pred)

        # ins
        ins_labels = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                                 for ins_labels_level_img, ins_ind_labels_level_img in
                                 zip(ins_labels_level, ins_ind_labels_level)], 0)
                      for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_label_list), zip(*ins_ind_label_list))]

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)
        num_ins = flatten_ins_ind_labels.int().sum()

        ins_ind_count_img = [
            [ins_ind_labels_level_img.flatten().int().sum()
                       for ins_ind_labels_level_img in ins_ind_labels_level]
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]

        ins_ind_index = []
        for i in range(len(ins_ind_labels)):
            ins_ind_index.append(torch.nonzero(ins_ind_labels[i])[:,0])
        

        # forward mask feature
        feature_pred = self.forward_mask_feat(feats)  

        # forward category heads
        cate_preds, size_preds, offset_preds, localmask_preds = multi_apply(self.forward_single_cat, 
                                                                        new_feats, 
                                                                        list(range(len(self.strides))),
                                                                        is_eval=False)


        attention_maps = [] 

        for j in range(len(self.seg_num_grids)):
            attention_maps_scale = multi_apply_custom(self.get_att_single, 
                                        [j for i in range(len(ins_ind_index[j]))],
                                        [feature_pred for i in range(len(ins_ind_index[j]))],
                                        ins_ind_index[j],
                                        is_eval=False)
            if len(attention_maps_scale):
                attention_maps_scale = torch.cat(attention_maps_scale, dim=0)
            else:
                attention_maps_scale = self.create_zeros_as(feature_pred, 
                                                [0,1,feature_pred.shape[-2],feature_pred.shape[-1]])
            attention_maps.append(attention_maps_scale)




        ins_preds_raw = multi_apply(self.forward_single_inst,  
                                        [feature_pred for i in range(len(new_feats))],
                                        attention_maps,
                                        ins_ind_count_img,
                                        list(range(len(self.strides))),
                                        featmap_sizes,
                                        is_eval=False)

        ins_preds = [ins[:,0,...] for ins in ins_preds_raw]
        '''for inst_level in ins_preds_raw:
            ins_preds.append(torch.cat([inst_level[i] for i in range(inst_level.shape[0])], dim=0))'''



        '''ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                                for ins_preds_level_img, ins_ind_labels_level_img in
                                zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in zip(ins_preds, zip(*ins_ind_label_list))]'''


        
        # dice loss
        loss_ins = []
        for input, target in zip(ins_preds, ins_labels):
            if input.size()[0] == 0:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(
            loss_ins=loss_ins,
            loss_cate=loss_cate)

    def solo_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               featmap_sizes=None):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):

            ins_label = torch.zeros([num_grid ** 2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = stride / 2

            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() < 10:
                   continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                # ins
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list

    def get_seg(self, feats, img_metas, cfg, rescale=None):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        #upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)

        feature_add_all_level = self.feature_convs[0](feats[0]) 
        for i in range(1,3):
            feature_add_all_level = feature_add_all_level + self.feature_convs[i](feats[i])
        feature_add_all_level = feature_add_all_level + self.feature_convs[3](feats[3])

        feature_pred = self.solo_mask(feature_add_all_level)  

        cate_preds = multi_apply_custom(self.forward_single_cat, 
                                new_feats, 
                                list(range(len(self.seg_num_grids))),
                                is_eval=True)




        num_levels = len(self.seg_num_grids)
        #print('num_levels: ' + '%d'%(num_levels))
        featmap_size_seg = feature_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):            
            '''cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]'''
            
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            #print(img_metas[img_id]['filename'])

            attention_maps = [] 
            cate_scores_list = []
            cate_labels_list = []
            strides_list = []

            for j in range(num_levels):
                cate_preds_level = cate_preds[j][img_id].view(-1, self.cate_out_channels).detach()
                inds_level = (cate_preds_level > cfg.score_thr)
                cate_scores_level = cate_preds_level[inds_level]
                #print('cate_scores_level: ' + '%d'%(len(cate_scores_level)))
                if len(cate_scores_level) == 0:
                    continue
                inds_level = inds_level.nonzero()
                cate_labels_level = inds_level[:, 1]

                strides = cate_preds_level.new_ones(inds_level.shape[0])
                strides *= self.strides[j]

                attention_maps_scale = multi_apply_custom(self.get_att_single, 
                                            [j for i in range(len(inds_level[:,0]))],
                                            [feature_pred[None,img_id] for i in range(len(inds_level[:,0]))],
                                            inds_level[:,0],
                                            is_eval=True)
                attention_maps_scale = torch.cat(attention_maps_scale, dim=0)

                attention_maps.append(attention_maps_scale)
                strides_list.append(strides)
                cate_labels_list.append(cate_labels_level)
                cate_scores_list.append(cate_scores_level)

            if len(attention_maps) == 0:
                #pdb.set_trace()
                result_list.append(None)
                continue

            seg_pred_list = multi_apply_custom(self.forward_single_inst, 
                                        [feature_pred[None,img_id] for i in range(len(new_feats))],
                                        attention_maps,
                                        list(range(len(new_feats))),
                                        is_eval=True)

            cate_scores_list = torch.cat(cate_scores_list, dim=0)
            cate_labels_list = torch.cat(cate_labels_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)
            strides_list = torch.cat(strides_list, dim=0)
            attention_maps_list = torch.cat(attention_maps, dim=0)

            #print(seg_pred_list.shape[0])

            result = self.get_seg_single(cate_scores_list, cate_labels_list, seg_pred_list, attention_maps_list,strides_list, featmap_size_seg, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_seg_single(self,
                       cate_scores,
                       cate_labels,
                       seg_preds,
                       attention_maps,
                       strides,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        seg_preds = seg_preds[:,0]
        attention_maps = attention_maps[:,0]

        # masks.
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()


        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        attention_maps = attention_maps[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        attention_maps = attention_maps[sort_inds, ...]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        attention_maps = attention_maps[keep, ...]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        attention_maps = attention_maps[sort_inds, ...]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr

        attention_maps = F.interpolate(attention_maps.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        attention_masks = F.interpolate(attention_maps,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        attention_masks = attention_masks > 0
        return seg_masks, cate_labels, cate_scores
        #return attention_masks, cate_labels, cate_scores
        