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
from itertools import chain
import pdb

INF = 1e8

from scipy import ndimage

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class CenterFocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(CenterFocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

'''class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss'''

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
  
    def forward(self, output, target):
        loss = F.l1_loss(pred, target, size_average=True)
        return loss

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

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
                 attention_size=16,
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
                 loss_offset=None,
                 loss_size=None,
                 loss_localmask=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SOLOAttHead, self).__init__()
        self.num_classes = num_classes
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.attention_size = attention_size
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
        self.loss_cate = CenterFocalLoss()
        self.loss_size = RegL1Loss()
        self.loss_offset = RegL1Loss()
        self.ins_loss_weight = loss_ins['loss_weight']
        self.offset_loss_weight = loss_offset['loss_weight']
        self.size_loss_weight = loss_size['loss_weight']
        self.localmask_loss_weight = loss_localmask['loss_weight']
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

        self.size_scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.strides))])
        self.offset_scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.strides))])

 
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

        normal_init(self.solo_mask.conv, std=0.01)
        self.inst_convs.init_weights()

    def create_zeros_as(self, x, shape):
        device = x.device
        target_type = x.dtype
        y = torch.zeros(shape, dtype=target_type, device=device)
        return y

    def get_att_single(self, featmap_size, stride, feature_pred, size_pred, offset_pred, localmask_pred, img_idx, position_idx, is_eval=False):
        device = feature_pred.device
        target_type = feature_pred.dtype
        N, c, h, w = feature_pred.shape

        attention = torch.zeros([1, 1, h, w], dtype=target_type, device=device)

        att_stride = 4.

        idx = position_idx % (featmap_size[0]*featmap_size[1])
        idx_h = idx // featmap_size[1]
        idx_w = idx % featmap_size[1]

        offset_w, offset_h = offset_pred[img_idx,:,idx_h,idx_w].detach().cpu().numpy()
        bbox_w, bbox_h = size_pred[img_idx,:,idx_h,idx_w].detach().cpu().numpy()
        localmask = localmask_pred[img_idx,:,idx_h,idx_w].view(1, 1, self.attention_size, self.attention_size)

        bbox_w_att = int(bbox_w/att_stride)
        bbox_h_att = int(bbox_h/att_stride)

        if bbox_w_att<=0 or bbox_h_att<=0:
            return attention,

        localmask = F.interpolate(localmask, size=(bbox_h_att, bbox_w_att), mode='bilinear', align_corners=True)

        center_w_att = int((idx_w*stride+offset_w)/att_stride)
        center_h_att = int((idx_h*stride+offset_h)/att_stride)

        w_min_raw = center_w_att - int(bbox_w/att_stride/2.)
        h_min_raw = center_h_att - int(bbox_h/att_stride/2.)
        w_max_raw = w_min_raw + bbox_w_att
        h_max_raw = h_min_raw + bbox_h_att

        if w_min_raw < 0:
            w_min = 0
            w_local_min = w_min - w_min_raw
        else:
            w_min = w_min_raw
            w_local_min = 0
        if w_max_raw > w:
            w_max = w
            w_local_max = bbox_w_att - (w_max_raw-w)
        else:
            w_max = w_max_raw
            w_local_max = bbox_w_att
        if (w_local_min>=bbox_w_att-1) or (w_local_max<=0) or (w_local_max<=w_local_min):
            return attention,

        if h_min_raw < 0:
            h_min = 0
            h_local_min = h_min - h_min_raw
        else:
            h_min = h_min_raw
            h_local_min = 0
        if h_max_raw > h:
            h_max = h
            h_local_max = bbox_h_att - (h_max_raw-h)
        else:
            h_max = h_max_raw
            h_local_max = bbox_h_att
        if (h_local_min>=bbox_h_att-1) or (h_local_max<=0) or (h_local_max<=h_local_min):
            return attention,

        attention[0,0,h_min:h_max,w_min:w_max] = localmask[0,0,h_local_min:h_local_max,w_local_min:w_local_max]

        return attention, 


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

        return inst_pred,


    def forward_single_cat(self, x, idx, is_eval=False):
        # cate head
        cate_feat = x
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)

        # size head
        size_feat = x
        for i, size_layer in enumerate(self.size_convs):
            size_feat = size_layer(size_feat)
        size_feat = F.relu(self.size_scales[idx](size_feat))

        # offset head
        offset_feat = x
        for i, offset_layer in enumerate(self.offset_convs):
            offset_feat = offset_layer(offset_feat)
        offset_feat = self.offset_scales[idx](offset_feat)

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
        new_feats = feats
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]

        # ins_ind_label_list is a list with only one element
        # ins_ind_label_list[0][0]: 1600,  ins_ind_label_list[0][1]: 1296
        # cate_label_list[0][0]: 40x40,  cate_label_list[0][1]: 36x36
        # ins_label_list[0][0]: 1600x288x192,  ins_label_list[0][0]:1296x288x192
        # len(ins_label_list) = len(cate_label_list) =len(ins_ind_label_list) = batch size
        # len(ins_label_list[0]) = number of feature levels
        ins_label_list, cate_label_list, ins_ind_index_list, offset_list, size_list, attention_list = multi_apply(
                self.solo_target_single,
                gt_bbox_list,
                gt_label_list,
                gt_mask_list,
                featmap_sizes=featmap_sizes)


        # ins
        # len(ins_labels) = number of feature levels
        # ins_labels[0].shape: [8, 272, 200]
        # ins_labels[1].shape: [10, 272, 200]
        # ins_labels[2].shape: [34, 136, 100]
        
        ins_labels = [torch.cat(list(ins_labels_level), 0) for ins_labels_level in zip(*ins_label_list)]
        attention_labels = [torch.cat(list(att_labels_level), 0) for att_labels_level in zip(*attention_list)]

        ins_ind_index = [list(chain(*ins_ind_index_level)) for ins_ind_index_level in zip(*ins_ind_index_list)]


        # num_ins is the number of positive samples
        num_ins = 0
        for index_level in ins_ind_index:
            num_ins += len(index_level)

        ins_ind_count_img = [[len(ins_ind_index_level_img) for ins_ind_index_level_img in ins_ind_index_level] 
                                for ins_ind_index_level in zip(*ins_ind_index_list)]

        ins_img_index = []
        for ins_ind_count_level in ins_ind_count_img:
            tmp = np.array([], dtype=np.int32)
            for p_idx, count in enumerate(ins_ind_count_level):
                tmp = np.concatenate([tmp,p_idx*np.ones(count).astype(np.int32)])
            ins_img_index.append(tmp)
        

        # forward mask feature
        feature_pred = self.forward_mask_feat(feats)  

        # forward category heads
        cate_preds, size_preds, offset_preds, localmask_preds = multi_apply(self.forward_single_cat, 
                                                                        new_feats, 
                                                                        list(range(len(self.strides))),
                                                                        is_eval=False)

        attention_maps = [] 

        for j in range(len(self.strides)):
            #pdb.set_trace()
            print(ins_img_index[j])
            attention_maps_scale, = multi_apply(self.get_att_single, 
                                        [featmap_sizes[j] for i in range(len(ins_ind_index[j]))],
                                        [self.strides[j] for i in range(len(ins_ind_index[j]))],
                                        [feature_pred for i in range(len(ins_ind_index[j]))],
                                        [size_preds[j] for i in range(len(ins_ind_index[j]))],
                                        [offset_preds[j] for i in range(len(ins_ind_index[j]))],
                                        [localmask_preds[j] for i in range(len(ins_ind_index[j]))],
                                        ins_img_index[j],
                                        ins_ind_index[j],
                                        is_eval=False)
            if len(attention_maps_scale):
                attention_maps_scale = torch.cat(attention_maps_scale, dim=0)
            else:
                attention_maps_scale = self.create_zeros_as(feature_pred, 
                                                [0,1,feature_pred.shape[-2],feature_pred.shape[-1]])
            attention_maps.append(attention_maps_scale)


        ins_preds_raw, = multi_apply(self.forward_single_inst,  
                                        [feature_pred for i in range(len(new_feats))],
                                        attention_maps,
                                        ins_ind_count_img,
                                        list(range(len(self.strides))),
                                        featmap_sizes,
                                        is_eval=False)

        ins_preds = [ins[:,0,...] for ins in ins_preds_raw]
        
        
        # dice loss
        loss_ins = []
        for input, target in zip(ins_preds, ins_labels):
            if input.size()[0] == 0:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # localmask dice loss
        localmask_preds = [
            localmask_pred.reshape(localmask_pred.shape[0], localmask_pred.shape[1], -1).permute(0,2,1)
            for localmask_pred in localmask_preds
        ]
        localmask_preds_select = []
        for level_idx in range(len(ins_ind_index_list[0])):
            tmp = []
            for img_idx in range(len(ins_ind_index_list)):
                indices = torch.tensor(ins_ind_index_list[img_idx][level_idx], dtype=torch.int64, device=localmask_preds[0].device)
                tmp.append(torch.index_select(localmask_preds[level_idx][img_idx], 0, indices))
            tmp = torch.cat(tmp)
            localmask_preds_select.append(tmp.reshape(tmp.shape[0], self.attention_size, self.attention_size))

        loss_attention = []
        for input, target in zip(localmask_preds_select, attention_labels):
            if input.size()[0] == 0:
                continue
            input = torch.sigmoid(input)
            loss_attention.append(dice_loss(input, target))
        loss_attention = torch.cat(loss_attention).mean()
        loss_attention = loss_attention * self.localmask_loss_weight


        # cate
        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        cate_labels = [
            cate_label.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_label in cate_label_list
        ]
        flatten_cate_labels = torch.cat(cate_labels)
        loss_cate = self.loss_cate(flatten_cate_preds.sigmoid_(), flatten_cate_labels)

        # offset loss, size loss
        offset_preds = [
            offset_pred.reshape(offset_pred.shape[0], offset_pred.shape[1], -1).permute(0,2,1)
            for offset_pred in offset_preds
        ]
        size_preds = [
            size_pred.reshape(size_pred.shape[0], size_pred.shape[1], -1).permute(0,2,1)
            for size_pred in size_preds
        ]
        offset_preds_select = []
        size_preds_select = []
        for img_idx, ins_index_img in enumerate(ins_ind_index_list):
            for level_idx, ins_index_img_level in enumerate(ins_index_img):
                indices = torch.tensor(ins_index_img_level, dtype=torch.int64, device=offset_preds[0].device)
                offset_preds_select.append(torch.index_select(offset_preds[level_idx][img_idx], 0, indices))
                size_preds_select.append(torch.index_select(size_preds[level_idx][img_idx], 0, indices))
        offset_preds_select = torch.cat(offset_preds_select)
        offset_gt_select = torch.cat([torch.cat(offset_img) for offset_img in offset_list])
        loss_offset = self.loss_offset(offset_preds_select, offset_gt_select)
        loss_offset = loss_offset * self.offset_loss_weight

        size_preds_select = torch.cat(size_preds_select)
        size_gt_select = torch.cat([torch.cat(size_img) for size_img in size_list])
        loss_size = self.loss_size(size_preds_select, size_gt_select)
        loss_size = loss_size * self.size_loss_weight

        return dict(
            loss_ins=loss_ins,
            loss_localmask=loss_attention,
            loss_cate=loss_cate,
            loss_offset=loss_offset,
            loss_size=loss_size)

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
        ins_ind_index_list = []
        offset_list = []
        size_list = []
        attention_list = []
        for (lower_bound, upper_bound), stride, featmap_size \
                in zip(self.scale_ranges, self.strides, featmap_sizes):

            #ins_label = torch.zeros([num_grid ** 2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            ins_label = []
            #cate_label = torch.zeros([self.cate_out_channels, featmap_size[0], featmap_size[1]], 
                #dtype=torch.float32, device=device)
            cate_label = np.zeros((self.cate_out_channels, featmap_size[0], featmap_size[1]), dtype=np.float32)
            ins_ind_index = []
            #ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)
            offsets = []
            sizes = []
            attentions = []

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(
                    torch.zeros([0, featmap_size[0]*2, featmap_size[1]*2], dtype=torch.uint8, device=device))
                cate_label_list.append(torch.tensor(cate_label, device=device))
                ins_ind_index_list.append(ins_ind_index)
                offset_list.append(torch.zeros([0, 2], dtype=torch.float32, device=device))
                size_list.append(torch.zeros([0, 2], dtype=torch.float32, device=device))
                attention_list.append(torch.zeros([0, self.attention_size, self.attention_size], dtype=torch.uint8, device=device))
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            output_stride = stride / 2

            for seg_mask, gt_label, gt_bbox in zip(gt_masks, gt_labels, gt_bboxes):
                if seg_mask.sum() < 10:
                   continue
                w_raw = gt_bbox[2] - gt_bbox[0]
                h_raw = gt_bbox[3] - gt_bbox[1]
                ct_raw = np.array([(gt_bbox[0] + gt_bbox[2]) / 2., (gt_bbox[1] + gt_bbox[3]) / 2.], dtype=np.float32)

                ct = ct_raw/stride
                ct_int = ct.astype(np.int32)
                offset_gt = (ct-ct_int)*stride

                w = w_raw/stride
                h = h_raw/stride

                #pdb.set_trace()

                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                draw_umich_gaussian(cate_label[gt_label], ct_int, radius)

                ins_ind_index.append(ct_int[1]*featmap_size[1]+ct_int[0])
                offsets.append(torch.tensor(offset_gt, device=device))
                sizes.append(torch.tensor([w_raw, h_raw], device=device))
                
                # ins
                seg_mask_resize = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask_resize = torch.Tensor(seg_mask_resize)
                ins_label_single = torch.zeros([featmap_size[0]*2, featmap_size[1]*2], dtype=torch.uint8, device=device)
                ins_label_single[:seg_mask_resize.shape[0], :seg_mask_resize.shape[1]] = seg_mask_resize
                ins_label.append(ins_label_single)

                attention_raw = seg_mask[int(gt_bbox[1]):int(gt_bbox[3]), int(gt_bbox[0]): int(gt_bbox[2])]
                attention = mmcv.imresize(attention_raw, (self.attention_size, self.attention_size))
                attention = torch.tensor(attention, dtype=torch.uint8, device=device)
                attentions.append(attention)

                #pdb.set_trace()
            assert len(ins_label) == len(ins_ind_index) == len(offsets) == len(sizes) == len(attentions)
            ins_label_list.append(torch.stack(ins_label, dim=0))
            cate_label_list.append(torch.tensor(cate_label, device=device))
            ins_ind_index_list.append(ins_ind_index)
            offset_list.append(torch.stack(offsets, dim=0))
            size_list.append(torch.stack(sizes, dim=0))
            #pdb.set_trace()
            attention_list.append(torch.stack(attentions, dim=0))
        return ins_label_list, cate_label_list, ins_ind_index_list, offset_list, size_list, attention_list

    def get_seg(self, feats, img_metas, cfg, rescale=None):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        #upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)

        feature_add_all_level = self.feature_convs[0](feats[0]) 
        for i in range(1,3):
            feature_add_all_level = feature_add_all_level + self.feature_convs[i](feats[i])
        feature_add_all_level = feature_add_all_level + self.feature_convs[3](feats[3])

        feature_pred = self.solo_mask(feature_add_all_level)  

        cate_preds, = multi_apply(self.forward_single_cat, 
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

                attention_maps_scale, = multi_apply(self.get_att_single, 
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

            seg_pred_list, = multi_apply(self.forward_single_inst, 
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
        