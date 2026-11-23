import torch
from torch import nn

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        bbox_overlaps, build_bbox_coder, reduce_mean)
from mmdet.models.builder import HEADS, build_roi_extractor
from .co_dino_head import CoDINOHead


@HEADS.register_module()
class LocalRefineCoDINOHead(CoDINOHead):
    """P1: refine final matching boxes using query and local spatial features."""

    def __init__(self, refine_topk=100, refine_channels=64,
                 refine_roi_size=7, refine_max_delta=0.5,
                 refine_enabled=True, refine_guard_weight=0.,
                 refine_context_scale=1.,
                 refine_featmap_strides=(4, 8, 16, 32, 64), **kwargs):
        super().__init__(**kwargs)
        if refine_topk < 1 or refine_max_delta <= 0:
            raise ValueError('refine_topk and refine_max_delta must be positive')
        if refine_channels < 8 or refine_channels % 8:
            raise ValueError('refine_channels must be a positive multiple of 8')
        self.refine_topk = refine_topk
        self.refine_max_delta = refine_max_delta
        self.refine_enabled = refine_enabled
        if not 1. <= refine_context_scale < float('inf'):
            raise ValueError('refine_context_scale must be finite and >= 1')
        self.refine_context_scale = refine_context_scale
        if refine_guard_weight < 0:
            raise ValueError('refine_guard_weight must be nonnegative')
        self.refine_guard_weight = refine_guard_weight
        self._guard_cache = None
        self.local_roi_extractor = build_roi_extractor(dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=refine_roi_size,
                           sampling_ratio=0),
            out_channels=self.embed_dims,
            featmap_strides=list(refine_featmap_strides), finest_scale=56))
        self.local_spatial = nn.Sequential(
            nn.Conv2d(self.embed_dims, refine_channels, 3, padding=1),
            nn.GroupNorm(8, refine_channels), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(refine_channels * refine_roi_size ** 2, self.embed_dims),
            nn.ReLU(inplace=True))
        self.local_fusion = nn.Sequential(
            nn.Linear(2 * self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True))
        self.local_delta = nn.Linear(self.embed_dims, 4)
        self.local_bbox_coder = build_bbox_coder(dict(
            type='DeltaXYWHBBoxCoder', target_stds=(1., 1., 1., 1.),
            clip_border=False))
        self._zero_refinement()

    def _zero_refinement(self):
        nn.init.zeros_(self.local_delta.weight)
        nn.init.zeros_(self.local_delta.bias)

    def init_weights(self):
        super().init_weights()
        self._zero_refinement()

    def refine_matching_boxes(self, feats, scores, boxes, queries, img_metas):
        batch, num_queries, _ = boxes.shape
        if num_queries == 0:
            return boxes
        # Rank queries, not flattened query/class pairs. Never use GT selection.
        indices = scores.detach().sigmoid().amax(dim=-1).topk(
            min(self.refine_topk, num_queries), dim=1).indices
        selected = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        selected_queries = queries.gather(
            1, indices[..., None].expand(-1, -1, queries.shape[-1]))
        factors = boxes.new_tensor([
            [meta['img_shape'][1], meta['img_shape'][0]] * 2
            for meta in img_metas])[:, None, :]
        pixel_boxes = bbox_cxcywh_to_xyxy(selected) * factors

        # DETR boxes are normalized to each resized image, NOT the padded batch.
        # Stop gradients through sampling coordinates; keep feature gradients.
        sample_boxes = pixel_boxes.detach().clamp(min=0)
        sample_boxes = torch.minimum(sample_boxes, factors)
        batch_ids = torch.arange(batch, device=boxes.device, dtype=boxes.dtype)
        batch_ids = batch_ids[:, None, None].expand(-1, indices.shape[1], 1)
        rois = torch.cat((batch_ids, sample_boxes), dim=-1).reshape(-1, 5)
        roi_feats = self.local_roi_extractor(
            feats[:self.local_roi_extractor.num_inputs], rois)
        spatial = self.local_spatial(roi_feats)
        if self.refine_context_scale > 1.:
            # Expand unclipped boxes, then clip to the same per-image bounds.
            center = (pixel_boxes[..., :2] + pixel_boxes[..., 2:]) * .5
            half_size = (pixel_boxes[..., 2:] - pixel_boxes[..., :2]) * (
                .5 * self.refine_context_scale)
            context_boxes = torch.cat((center - half_size, center + half_size), -1)
            context_boxes = torch.minimum(context_boxes.detach().clamp(min=0), factors)
            context_rois = torch.cat((batch_ids, context_boxes), -1).reshape(-1, 5)
            context_feats = self.local_roi_extractor(
                feats[:self.local_roi_extractor.num_inputs], context_rois)
            # Shared encoder and fixed fusion isolate context without extra parameters.
            spatial = .5 * (spatial + self.local_spatial(context_feats))
        fused = self.local_fusion(torch.cat((
            selected_queries.reshape(-1, self.embed_dims), spatial), dim=-1))
        deltas = self.local_delta(fused).tanh() * self.refine_max_delta
        decoded = self.local_bbox_coder.decode(pixel_boxes.reshape(-1, 4), deltas)
        refined = bbox_xyxy_to_cxcywh(decoded.reshape_as(pixel_boxes)) / factors
        # Keep the existing normalized cxcywh output contract. Inference applies
        # the usual xyxy clipping/rescaling downstream, just as in the baseline.
        refined = refined.clamp(min=0., max=1.)
        return boxes.scatter(1, indices[..., None].expand(-1, -1, 4), refined)

    def forward(self, mlvl_feats, img_metas, dn_label_query=None,
                dn_bbox_query=None, attn_mask=None):
        self._guard_cache = None
        if not self.refine_enabled and self.training:
            raise RuntimeError('refine_enabled=False is an inference-only ablation')
        scores, boxes, topk_score, topk_anchor, enc_feats = super().forward(
            mlvl_feats, img_metas, dn_label_query, dn_bbox_query, attn_mask)
        if not self.refine_enabled:
            return scores, boxes, topk_score, topk_anchor, enc_feats
        pad_size = 0 if dn_label_query is None else dn_label_query.size(1)
        refined = self.refine_matching_boxes(
            enc_feats, scores[-1, :, pad_size:], boxes[-1, :, pad_size:],
            self._last_decoder_hs[-1, :, pad_size:], img_metas)
        if self.training and self.refine_guard_weight > 0:
            self._guard_cache = (scores[-1, :, pad_size:].detach(),
                                 boxes[-1, :, pad_size:].detach(), refined)
        final_boxes = torch.cat((boxes[-1, :, :pad_size], refined), dim=1)
        boxes = torch.cat((boxes[:-1], final_boxes.unsqueeze(0)), dim=0)
        # The parent loss consumes these boxes for Hungarian matching, QFL IoU
        # targets, L1 and GIoU. DN/earlier layers and encoder outputs are intact.
        return scores, boxes, topk_score, topk_anchor, enc_feats

    def refinement_guard_loss(self, scores, before, after, gt_bboxes,
                              gt_labels, img_metas, gt_bboxes_ignore=None):
        targets = self.get_targets(list(scores), list(after.detach()), gt_bboxes,
                                   gt_labels, img_metas, gt_bboxes_ignore)
        labels = torch.cat(targets[0])
        selected = torch.zeros_like(scores[..., 0], dtype=torch.bool)
        selected.scatter_(1, scores.sigmoid().amax(-1).topk(
            min(self.refine_topk, scores.shape[1]), dim=1).indices, True)
        mask = selected.flatten() & (labels >= 0) & (labels < self.num_classes)
        factors = after.new_tensor([[m['img_shape'][1], m['img_shape'][0]] * 2
                                    for m in img_metas])[:, None, :]
        factors = factors.expand_as(after).reshape(-1, 4)[mask]
        gt = bbox_cxcywh_to_xyxy(torch.cat(targets[2])[mask].float()) * factors
        old = bbox_cxcywh_to_xyxy(before.detach().reshape(-1, 4)[mask].float()) * factors
        new = bbox_cxcywh_to_xyxy(after.reshape(-1, 4)[mask].float()) * factors
        old_iou = bbox_overlaps(old, gt, is_aligned=True).detach()
        new_iou = bbox_overlaps(new, gt, is_aligned=True)
        # Normalize by selected positives, not only examples with active penalty.
        normalizer = reduce_mean(after.new_tensor(float(mask.sum()))).clamp(min=1)
        return (self.refine_guard_weight * (old_iou - new_iou).relu().sum()
                / normalizer + after.sum() * 0.)

    def loss(self, all_cls_scores, all_bbox_preds, enc_topk_scores,
             enc_topk_anchors, enc_outputs, gt_bboxes_list, gt_labels_list,
             img_metas, dn_meta=None, gt_bboxes_ignore=None):
        cache, self._guard_cache = self._guard_cache, None
        losses = super().loss(all_cls_scores, all_bbox_preds, enc_topk_scores,
                              enc_topk_anchors, enc_outputs, gt_bboxes_list,
                              gt_labels_list, img_metas, dn_meta, gt_bboxes_ignore)
        if self.refine_guard_weight > 0:
            if cache is None:
                raise RuntimeError('Guard loss requires the matching training forward')
            losses['loss_refine_guard'] = self.refinement_guard_loss(
                *cache, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore)
        return losses
