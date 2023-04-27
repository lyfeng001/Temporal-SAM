# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self,option):
        super(ModelBuilder, self).__init__()

        if option:
            # build backbone
            self.backbone = get_backbone(cfg.KD.BACKBONE_TYPE,
                                         **cfg.BACKBONE.KWARGS)  # 将cfg文件中的backbone类型参数以及其他关键字参数输入，其中关键字参数为字典类型，通过**解包

            # build adjust layer
            if cfg.ADJUST.ADJUST:
                self.neck = get_neck(cfg.KD.ADJUST.TYPE,
                                     **cfg.ADJUST.KWARGS)

            # build rpn head
            self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                         **cfg.RPN.KWARGS)

            # build mask head
            if cfg.MASK.MASK:
                self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                               **cfg.MASK.KWARGS)

                if cfg.REFINE.REFINE:
                    self.refine_head = get_refine_head(cfg.REFINE.TYPE)  # 下面的这些模块也与backbone同理

        else:

            # build backbone
            self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                         **cfg.BACKBONE.KWARGS)   #将cfg文件中的backbone类型参数以及其他关键字参数输入，其中关键字参数为字典类型，通过**解包

            # build adjust layer
            if cfg.ADJUST.ADJUST:
                self.neck = get_neck(cfg.ADJUST.TYPE,
                                     **cfg.ADJUST.KWARGS)

            # build rpn head
            self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                         **cfg.RPN.KWARGS)

            # build mask head
            if cfg.MASK.MASK:
                self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                               **cfg.MASK.KWARGS)

                if cfg.REFINE.REFINE:
                    self.refine_head = get_refine_head(cfg.REFINE.TYPE)         #下面的这些模块也与backbone同理

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc, cls_feature, loc_feature = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data, teacher_model=None):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc, cls_feature, loc_feature = self.rpn_head(zf, xf)


        tea_zf = teacher_model.backbone(template)
        tea_xf = teacher_model.backbone(search)
        tea_cls, tea_loc, tea_cls_feature, tea_loc_feature = teacher_model.rpn_head(tea_zf, tea_xf)

        ### kd loss
        soft_loss = nn.KLDivLoss(reduction='batchmean')

        kd_loss = soft_loss(F.softmax(cls/Tem,dim=1),F.softmax(tea_cls/Tem,dim=1))+ \
                    soft_loss(F.softmax(loc/Tem,dim=1),F.softmax(tea_loc/Tem,dim=1))

        ### corr_loss
        L2_loss = nn.MSELoss()
        corr_loss = L2_loss(cls_feature,tea_cls_feature) + L2_loss(loc_feature,tea_loc_feature)


        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CORR_WEIGHT * corr_loss + cfg.TRAIN.KD_WEIGHT * kd_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        outputs['kd_loss'] = kd_loss
        outputs['corr_loss'] = corr_loss




        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
