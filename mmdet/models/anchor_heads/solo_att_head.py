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

class _ConvNd_Group(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 bias, padding_mode):
        super(_ConvNd_Group, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd_Group, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class Conv2d_Group(_ConvNd_Group):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_Group, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), bias, padding_mode)

    def conv2d_forward(self, input, group, weight, bias):
        weight_cat = torch.cat([weight for i in range(group)])
        bias_cat = torch.cat([bias for i in range(group)])
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight_cat, bias_cat, self.stride,
                            _pair(0), self.dilation, group)
        return F.conv2d(input, weight_cat, bias_cat, self.stride,
                        self.padding, self.dilation, group)

    def forward(self, input, group):
        return self.conv2d_forward(input, group, self.weight, self.bias)


class MaskAttModule(nn.Module):
    def __init__(self, mask_in, out_channels):
        super(MaskAttModule, self).__init__()
        self.mask_conv = nn.Conv2d(mask_in, out_channels, 3, padding=1)
        self.att_conv = Conv2d_Group(1, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, mask_feat, attention, group):
        # mask_feat: N,C,H,W (C is the number of channels)
        # attention: N,Ins,H,W (Ins is the number of instances)
        mask_output = self.mask_conv(mask_feat)
        att_output = self.att_conv(attention, group)
        output = torch.cat([mask_output for i in range(group)], dim=1) + att_output
        return self.relu(output)


class InstanceHead(nn.Module):
    def __init__(self, mask_in, out_channels, num_conv):
        super(InstanceHead, self).__init__()
        self.num_conv = num_conv
        self.mask_att_module = MaskAttModule(mask_in, out_channels)
        for i in range(num_conv):
            setattr(self, 'conv_%d'%(i), Conv2d_Group(out_channels, out_channels, 3, padding=1))
            setattr(self, 'relu_%d'%(i), nn.ReLU())
        self.conv_final = Conv2d_Group(out_channels, 1, 3, padding=1)

    def forward(self, mask_feat, attention, group):
        # mask_feat: N,C,H,W (C is the number of channels)
        # attention: N,Ins,H,W (Ins is the number of instances)
        x = self.mask_att_module(mask_feat, attention, group)
        for i in range(self.num_conv):
            conv_layer = getattr(self, 'conv_%d'%(i))
            relu_layer = getattr(self, 'relu_%d'%(i))
            x = conv_layer(x, group)
            x = relu_layer(x)
        x = self.conv_final(x, group)
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
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 gauss_ranges=(48, 96, 192, 384, 768),
                 sigma=0.4,
                 num_grids=None,
                 cate_down_pos=0,
                 with_deform=False,
                 loss_ins=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SOLOAttHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.inst_feat_channels = inst_feat_channels
        self.num_inst_convs = inst_convs
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.gauss_ranges = gauss_ranges
        self.with_deform = with_deform
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        self._init_fix_gauss_att()

    def _init_fix_gauss_att(self):
        from scipy.stats import multivariate_normal
        from skimage.transform import resize
        x, y = np.mgrid[-1:1:.005, -1:1:.005]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        rv = multivariate_normal([0., 0.], [[0.5, 0.], [0., 0.5]])
        dist = 3*rv.pdf(pos)
        self.att_pyramid = []
        for height in self.gauss_ranges:
            self.att_pyramid.append(torch.tensor(resize(dist, (height+1, height+1))).cuda())


    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.feature_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        self.inst_convs = InstanceHead(self.seg_feat_channels, self.inst_feat_channels, self.num_inst_convs)
        
        # mask feature     
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
        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)
        self.solo_mask = ConvModule(
            self.seg_feat_channels, self.seg_feat_channels, 1, padding=0, norm_cfg=norm_cfg, bias=norm_cfg is None)

 
    def init_weights(self):
        #TODO: init for feat_conv
        for m in self.feature_convs:
            s=len(m)
            for i in range(s):
                if i%2 == 0:
                    normal_init(m[i].conv, std=0.01)
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)

    def get_att_single(self, scale_idx, feature_pred, idx, is_eval=False):
        device = feature_pred.device
        target_type = feature_pred.dtype
        N, c, h, w = feature_pred.shape
        num_grid = self.seg_num_grids[scale_idx]
        att_template = self.att_pyramid[scale_idx]
        h_att, w_att = att_template.shape
        h_att_half = (h_att-1)//2
        w_att_half = (w_att-1)//2

        attention = torch.zeros([N, 1, h, w], dtype=target_type, device=device)
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

        for i in range(N):
            attention[i,0,h_min:h_max,w_min:w_max] = att_template[h_att_min:h_att_max,w_att_min:w_att_max]

        return attention


    '''def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (feats[0].shape[-2], feats[0].shape[-3])

        feature_add_all_level = self.feature_convs[0](feats[0]) 
        for i in range(1,3):
            feature_add_all_level = feature_add_all_level + self.feature_convs[i](feats[i])
        feature_add_all_level = feature_add_all_level + self.feature_convs[3](feats[3])
        
        feature_pred = self.solo_mask(feature_add_all_level)  
        attention_maps = [] 

        for j in range(len(self.seg_num_grids)):
            attention_maps_scale = multi_apply_custom(self.get_att_single, 
                                        [j for i in range(self.seg_num_grids[j]**2)],
                                        [feature_pred for i in range(self.seg_num_grids[j]**2)],
                                        list(range(self.seg_num_grids[j]**2)),
                                        eval=eval)
            attention_maps_scale = torch.cat(attention_maps_scale, dim=1)
            attention_maps.append(attention_maps_scale)

        ins_pred, cate_pred = multi_apply(self.forward_single, 
                                        new_feats, 
                                        [feature_pred for i in range(len(new_feats))],
                                        attention_maps,
                                        featmap_sizes,
                                        list(range(len(self.seg_num_grids))),
                                        eval=eval)

        return ins_pred, cate_pred'''

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'), 
                feats[1], 
                feats[2], 
                feats[3], 
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def forward_single(self, x, mask_feat, attention, featmap_size, idx, is_eval=False):
        cate_feat = x
        # cate branch
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = cate_layer(cate_feat)

        cate_pred = self.solo_cate(cate_feat)

        if attention.shape[1]:
            inst_pred = self.inst_convs(mask_feat, attention, attention.shape[1])
            if is_eval:
                inst_pred = inst_pred.sigmoid()
            else:
                inst_pred = F.interpolate(inst_pred, size=(featmap_size[0]*2,featmap_size[1]*2), mode='bilinear')

        else:
            device = mask_feat.device
            target_type = mask_feat.dtype
            N, c, h, w = mask_feat.shape
            inst_pred = torch.zeros([N, 0, h, w], dtype=target_type, device=device)

        if is_eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return inst_pred, cate_pred

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
        


        feature_add_all_level = self.feature_convs[0](feats[0]) 
        for i in range(1,3):
            feature_add_all_level = feature_add_all_level + self.feature_convs[i](feats[i])
        feature_add_all_level = feature_add_all_level + self.feature_convs[3](feats[3])

        feature_pred = self.solo_mask(feature_add_all_level)  
        attention_maps = [] 

        pdb.set_trace()

        for j in range(len(self.seg_num_grids)):
            attention_maps_scale = multi_apply_custom(self.get_att_single, 
                                        [j for i in range(len(ins_ind_index[j]))],
                                        [feature_pred for i in range(len(ins_ind_index[j]))],
                                        ins_ind_index[j],
                                        is_eval=False)
            if len(attention_maps_scale):
                attention_maps_scale = torch.cat(attention_maps_scale, dim=1)
            else:
                device = feature_pred.device
                target_type = feature_pred.dtype
                N, c, h, w = feature_pred.shape
                attention_maps_scale = torch.zeros([N, 0, h, w], dtype=target_type, device=device)
            attention_maps.append(attention_maps_scale)



        ins_preds_raw, cate_preds = multi_apply(self.forward_single, 
                                        new_feats, 
                                        [feature_pred for i in range(len(new_feats))],
                                        attention_maps,
                                        featmap_sizes,
                                        list(range(len(self.seg_num_grids))),
                                        is_eval=False)

        ins_preds = []
        for inst_level in ins_preds_raw:
            ins_preds.append(torch.cat([inst_level[i] for i in range(inst_level.shape[0])], dim=0))



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

    def get_seg(self, seg_preds, cate_preds, img_metas, cfg, rescale=None):
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = [
                seg_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):
        assert len(cate_preds) == len(seg_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
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
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores
        