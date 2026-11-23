# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from .assign_result import AssignResult
from .max_iou_assigner import MaxIoUAssigner


@BBOX_ASSIGNERS.register_module()
class DynamicAssigner(MaxIoUAssigner):
    """IoU-aware assigner used by the MOQAM DI-RPN branch.

    The paper raises the positive IoU threshold smoothly with object scale:

        t_img = max(t_b, t_b + k * (sqrt(w * h) / A_s) ** p)

    In practice the threshold is capped by ``pos_iou_thr`` so large objects do
    not become impossible positives. This assigner is only used during
    training; inference behavior is unchanged.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 low_quality_iou_thr=None,
                 base_pos_iou_thr=0.55,
                 k=0.15,
                 p=0.2,
                 smallest_instance_scale=None,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        super().__init__(
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr,
            min_pos_iou=min_pos_iou,
            gt_max_assign_all=gt_max_assign_all,
            ignore_iof_thr=ignore_iof_thr,
            ignore_wrt_candidates=ignore_wrt_candidates,
            match_low_quality=match_low_quality,
            gpu_assign_thr=gpu_assign_thr,
            iou_calculator=iou_calculator)
        self.low_quality_iou_thr = low_quality_iou_thr
        self.base_pos_iou_thr = base_pos_iou_thr
        self.dynamic_k = k
        self.growth_exponent = p
        self.smallest_instance_scale = smallest_instance_scale

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt boxes using per-GT dynamic positive thresholds."""
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False

        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        dynamic_thresholds = self._dynamic_thresholds(gt_bboxes)
        assign_result = self.assign_wrt_overlaps_dynamic(
            overlaps, dynamic_thresholds, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def _dynamic_thresholds(self, gt_bboxes):
        if gt_bboxes.numel() == 0:
            return gt_bboxes.new_zeros((0, ))
        widths = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1.0)
        heights = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1.0)
        scales = torch.sqrt(widths * heights)
        if self.smallest_instance_scale is None:
            denom = scales.detach().min().clamp(min=1.0)
        else:
            denom = scales.new_tensor(float(self.smallest_instance_scale)).clamp(
                min=1.0)
        thresholds = self.base_pos_iou_thr + self.dynamic_k * (
            scales / denom).pow(self.growth_exponent)
        thresholds = thresholds.clamp(
            min=self.base_pos_iou_thr, max=self.pos_iou_thr)
        return thresholds

    def assign_wrt_overlaps_dynamic(self,
                                    overlaps,
                                    dynamic_thresholds,
                                    gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        pos_thr_per_bbox = dynamic_thresholds[argmax_overlaps]
        pos_inds = max_overlaps >= pos_thr_per_bbox
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            low_quality_thr = self.low_quality_iou_thr
            if low_quality_thr is None:
                low_quality_thr = self.min_pos_iou
            for i in range(num_gts):
                if gt_max_overlaps[i] >= low_quality_thr:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module()
class DistributionAwareDynamicAssigner(DynamicAssigner):
    """Full distribution-aware DI-RPN assigner.

    This extends the recovered DI-RPN ``DynamicAssigner`` with a Gaussian/KL
    candidate ranking term. Each gt box and candidate anchor/proposal is modeled
    as a diagonal 2D Gaussian centered at the box center. Candidates close to a
    gt in KL space receive higher distribution quality and can become positives
    through a per-gt top-k path, which is especially useful for small findings
    where IoU alone is brittle.
    """

    def __init__(self,
                 *args,
                 topk=9,
                 beta=0.9,
                 distribution_min_iou=0.1,
                 distribution_weight=0.25,
                 gaussian_scale=6.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.topk = topk
        self.beta = beta
        self.distribution_min_iou = distribution_min_iou
        self.distribution_weight = distribution_weight
        self.gaussian_scale = gaussian_scale

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False

        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        dynamic_thresholds = self._dynamic_thresholds(gt_bboxes)
        dist_quality, topk_mask = self._distribution_quality(
            gt_bboxes, bboxes, overlaps)
        assign_result = self.assign_wrt_distribution_dynamic(
            overlaps, dynamic_thresholds, dist_quality, topk_mask, gt_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def _bbox_to_gaussian(self, boxes):
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1.0)
        std = (wh / float(self.gaussian_scale)).clamp(min=1.0)
        var = std.pow(2)
        return centers, var

    def _distribution_quality(self, gt_bboxes, bboxes, overlaps):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        if num_gts == 0 or num_bboxes == 0:
            return overlaps.new_zeros((num_gts, num_bboxes)), overlaps.new_zeros(
                (num_gts, num_bboxes), dtype=torch.bool)

        gt_centers, gt_var = self._bbox_to_gaussian(gt_bboxes)
        box_centers, box_var = self._bbox_to_gaussian(bboxes)

        diff = gt_centers[:, None, :] - box_centers[None, :, :]
        gt_var = gt_var[:, None, :]
        box_var = box_var[None, :, :].clamp(min=1e-6)
        kl = 0.5 * (
            torch.log(box_var / gt_var.clamp(min=1e-6)).sum(dim=-1) +
            ((gt_var + diff.pow(2)) / box_var).sum(dim=-1) - 2.0)
        kl = kl.clamp(min=0.0)

        valid = overlaps >= 0
        quality = torch.exp(-float(self.beta) * kl)
        quality = quality * valid.float()

        topk_mask = torch.zeros_like(valid)
        k = min(int(self.topk), num_bboxes)
        if k > 0:
            masked_kl = kl.masked_fill(~valid, float('inf'))
            _, topk_inds = torch.topk(
                -masked_kl, k=k, dim=1, largest=True, sorted=False)
            topk_mask.scatter_(1, topk_inds, True)
            topk_mask = topk_mask & valid
        return quality, topk_mask

    def assign_wrt_distribution_dynamic(self,
                                        overlaps,
                                        dynamic_thresholds,
                                        dist_quality,
                                        topk_mask,
                                        gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        max_overlaps, _ = overlaps.max(dim=0)
        fused_overlaps = overlaps + float(self.distribution_weight) * dist_quality
        fused_overlaps = fused_overlaps.masked_fill(overlaps < 0, -1)
        _, argmax_fused = fused_overlaps.max(dim=0)
        matched_overlaps = overlaps[argmax_fused,
                                    torch.arange(num_bboxes, device=overlaps.device)]

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        pos_thr_per_bbox = dynamic_thresholds[argmax_fused]
        dynamic_pos = matched_overlaps >= pos_thr_per_bbox
        distribution_pos = (
            topk_mask[argmax_fused,
                      torch.arange(num_bboxes, device=overlaps.device)] &
            (matched_overlaps >= float(self.distribution_min_iou)))
        pos_inds = dynamic_pos | distribution_pos
        assigned_gt_inds[pos_inds] = argmax_fused[pos_inds] + 1

        if self.match_low_quality:
            low_quality_thr = self.low_quality_iou_thr
            if low_quality_thr is None:
                low_quality_thr = self.min_pos_iou
            for i in range(num_gts):
                gt_overlaps = overlaps[i, :]
                gt_max_overlap = gt_overlaps.max()
                if gt_max_overlap >= low_quality_thr:
                    if self.gt_max_assign_all:
                        max_iou_inds = gt_overlaps == gt_max_overlap
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_overlaps.argmax()] = i + 1

                dist_candidates = (topk_mask[i, :] &
                                   (gt_overlaps >=
                                    float(self.distribution_min_iou)))
                if dist_candidates.any():
                    assigned_gt_inds[dist_candidates] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module()
class ConflictAwareDistributionDynamicAssigner(
        DistributionAwareDynamicAssigner):
    """Conservative distribution assignment with deterministic GT conflicts."""

    def __init__(self, *args, distribution_quality_thr=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.distribution_quality_thr = distribution_quality_thr

    def assign_wrt_distribution_dynamic(self,
                                        overlaps,
                                        dynamic_thresholds,
                                        dist_quality,
                                        topk_mask,
                                        gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        assigned_gt_inds = overlaps.new_full(
            (num_bboxes, ), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            assigned_labels = None if gt_labels is None else \
                assigned_gt_inds.new_full((num_bboxes, ), -1)
            return AssignResult(
                num_gts, assigned_gt_inds, max_overlaps,
                labels=assigned_labels)

        bbox_inds = torch.arange(num_bboxes, device=overlaps.device)
        max_overlaps, _ = overlaps.max(dim=0)
        fused_overlaps = overlaps + float(self.distribution_weight) * dist_quality
        fused_overlaps = fused_overlaps.masked_fill(overlaps < 0, -1)
        _, best_gt = fused_overlaps.max(dim=0)
        matched_overlaps = overlaps[best_gt, bbox_inds]
        matched_quality = dist_quality[best_gt, bbox_inds]

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) &
                             (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0]) &
                             (max_overlaps < self.neg_iou_thr[1])] = 0

        dynamic_pos = matched_overlaps >= dynamic_thresholds[best_gt]
        distribution_pos = (
            topk_mask[best_gt, bbox_inds] &
            (matched_overlaps >= float(self.distribution_min_iou)) &
            (matched_quality >= float(self.distribution_quality_thr)))
        pos_inds = dynamic_pos | distribution_pos
        assigned_gt_inds[pos_inds] = best_gt[pos_inds] + 1

        if self.match_low_quality:
            low_quality_thr = self.low_quality_iou_thr
            if low_quality_thr is None:
                low_quality_thr = self.min_pos_iou
            gt_max_overlaps = overlaps.max(dim=1)[0]
            eligible_gt = gt_max_overlaps >= low_quality_thr
            force_mask = overlaps == gt_max_overlaps[:, None]
            force_mask &= eligible_gt[:, None]
            forced_candidates = force_mask.any(dim=0)
            if forced_candidates.any():
                forced_scores = fused_overlaps.masked_fill(~force_mask, -1)
                forced_best_gt = forced_scores.argmax(dim=0)
                assigned_gt_inds[forced_candidates] = (
                    forced_best_gt[forced_candidates] + 1)

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze(1)
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
