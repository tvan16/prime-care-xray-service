"""Knowledge distillation for the pruned Co-DETR variants.

Why distillation, and why this particular form, is settled by the diagnostics
in NOTE.md rather than by convention:

  * The oracle experiment puts every architecture tried so far at the same
    ceiling (+10.17 control, +10.28 for v3). The pruned student is therefore
    *already capable* of the teacher's accuracy -- its boxes are as good; it
    just scores them 1.17 AP worse.
  * Phase 10 showed the gap is not recoverable by re-scoring frozen features
    after the fact, which is exactly what soft targets inside training avoid.

So the transfer target is the teacher's *ranking* signal: its soft class
scores on the detections both models already agree on. Box regression is
distilled too, but with a smaller weight, because localisation is not where
the measured gap lives.

Teacher and student do not share a backbone here (the student drops the
stride-4 level and half the encoder), so feature-map distillation has no
aligned levels to match. The decoder outputs, on the other hand, line up
exactly: both produce `num_query` predictions over the same label set. Those
are matched one-to-one with the Hungarian algorithm on box cost alone -- we
want to pair up the same physical detection in both models, not to re-solve
the detection problem.

The teacher is held in a plain list so `nn.Module` never registers it: it
stays out of `state_dict`, out of the optimiser, and out of the saved
checkpoint, which keeps the student's checkpoints the same size as any other
run.
"""
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from scipy.optimize import linear_sum_assignment

from mmdet.core import bbox_cxcywh_to_xyxy
from mmdet.models.builder import DETECTORS, build_detector
from mmdet.models.losses import GIoULoss

from .co_detr import CoDETR


@DETECTORS.register_module()
class CoDETRDistill(CoDETR):
    """CoDETR student supervised by a frozen CoDETR teacher.

    Args:
        teacher_config (str): config file of the teacher.
        teacher_checkpoint (str): checkpoint to load into the teacher.
        distill (dict): see `DEFAULT_DISTILL` for the keys and defaults.
        All other arguments are passed straight through to `CoDETR`.
    """

    DEFAULT_DISTILL = dict(
        cls_weight=2.0,       # soft-BCE on matched class logits
        bbox_weight=1.0,      # L1 on matched boxes
        giou_weight=0.5,      # GIoU on matched boxes
        topk=100,             # teacher detections kept per image
        score_thr=0.05,       # ... and the minimum teacher confidence
        match_l1_cost=5.0,    # Hungarian cost: L1 on cxcywh
        match_giou_cost=2.0,  # Hungarian cost: 1 - GIoU
        conf_weighting=True,  # weight each pair by the teacher's confidence
        layers='last',        # 'last' or 'all' decoder layers
        # Suppression of the queries that were NOT matched to a teacher
        # detection. Without it the loss only ever pushes scores up, which
        # breaks the one-to-one suppression DETR relies on and multiplies
        # duplicates (measured: 17,640 -> 40,237, duplicate_cost 0.72 -> 4.85).
        suppress_weight=2.0,
    )

    def __init__(self,
                 *args,
                 teacher_config=None,
                 teacher_checkpoint=None,
                 distill=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher_config is not None, 'teacher_config is required'

        cfg = dict(self.DEFAULT_DISTILL)
        cfg.update(distill or {})
        assert cfg['layers'] in ('last', 'all')
        self.distill_cfg = cfg

        teacher = self._build_teacher(teacher_config, teacher_checkpoint)
        # A plain list, so nn.Module does not register it. The teacher then
        # stays out of state_dict / parameters() / the saved checkpoint.
        self._teacher_holder = [teacher]
        self._teacher_on_device = False
        self._giou = GIoULoss(reduction='none')

    # ---------------------------------------------------------------- teacher

    @staticmethod
    def _build_teacher(teacher_config, teacher_checkpoint):
        cfg = Config.fromfile(teacher_config)
        cfg.model.pop('pretrained', None)
        cfg.model.pop('init_cfg', None)
        teacher = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        if teacher_checkpoint:
            load_checkpoint(teacher, teacher_checkpoint, map_location='cpu')
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    @property
    def teacher(self):
        return self._teacher_holder[0]

    def train(self, mode=True):
        """Keep the teacher in eval mode whatever the student does."""
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def _teacher_outputs(self, img, img_metas):
        """Teacher class logits and boxes for the matching queries.

        Returns tensors shaped [n_dec, bs, num_query, num_classes] and
        [n_dec, bs, num_query, 4] (normalised cxcywh).
        """
        teacher = self.teacher
        if not self._teacher_on_device:
            teacher.to(img.device)
            self._teacher_on_device = True
        teacher.eval()

        x = teacher.extract_feat(img, img_metas)
        # No denoising queries at inference, so the matching part is all of it.
        outs = teacher.query_head(x, img_metas)
        return outs[0], outs[1]

    # ------------------------------------------------------------------ train

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, **kwargs):
        losses = super().forward_train(img, img_metas, gt_bboxes, gt_labels,
                                       **kwargs)

        stash = getattr(self.query_head, '_last_train_outs', None)
        if stash is None:
            # query_head disabled: nothing aligned to distil against.
            return losses
        s_cls_all, s_box_all, dn_meta = stash
        s_cls, s_box, _, _ = self.query_head.extract_dn_outputs(
            s_cls_all, s_box_all, dn_meta)

        t_cls, t_box = self._teacher_outputs(img, img_metas)

        losses.update(self._distill_losses(s_cls, s_box, t_cls, t_box))
        return losses

    def _distill_losses(self, s_cls, s_box, t_cls, t_box):
        cfg = self.distill_cfg
        layer_ids = range(s_cls.size(0)) if cfg['layers'] == 'all' else [-1]

        cls_terms, box_terms, giou_terms, sup_terms = [], [], [], []
        for lvl in layer_ids:
            c, b, g, u = self._distill_one_layer(s_cls[lvl], s_box[lvl],
                                                 t_cls[lvl], t_box[lvl])
            cls_terms.append(c)
            box_terms.append(b)
            giou_terms.append(g)
            sup_terms.append(u)

        n = max(len(cls_terms), 1)
        out = {}
        if cfg['cls_weight'] > 0:
            out['loss_distill_cls'] = cfg['cls_weight'] * sum(cls_terms) / n
        if cfg['bbox_weight'] > 0:
            out['loss_distill_bbox'] = cfg['bbox_weight'] * sum(box_terms) / n
        if cfg['giou_weight'] > 0:
            out['loss_distill_giou'] = cfg['giou_weight'] * sum(giou_terms) / n
        if cfg['suppress_weight'] > 0:
            out['loss_distill_sup'] = (
                cfg['suppress_weight'] * sum(sup_terms) / n)
        return out

    def _distill_one_layer(self, s_cls, s_box, t_cls, t_box):
        """One decoder layer, summed over the batch.

        s_cls/t_cls: [bs, num_query, num_classes]; s_box/t_box: [bs, num_query, 4].
        """
        cfg = self.distill_cfg
        device = s_cls.device
        zero = s_cls.sum() * 0.0

        cls_total, box_total, giou_total, sup_total = zero, zero, zero, zero
        n_pairs = 0

        for i in range(s_cls.size(0)):
            keep = self._select_teacher(t_cls[i])
            if keep.numel() == 0:
                continue
            tc, tb = t_cls[i][keep], t_box[i][keep]

            col = self._match(s_box[i], tb)
            if col is None:
                continue
            sc, sb = s_cls[i][col], s_box[i][col]

            # Teacher confidence per pair, used as the pair's weight.
            conf = tc.sigmoid().max(-1).values
            w = conf if cfg['conf_weighting'] else torch.ones_like(conf)
            w_sum = w.sum().clamp(min=1e-6)

            # Soft BCE: the teacher's per-class probability is the target.
            # This is the ranking signal the oracle analysis pointed at.
            soft = tc.sigmoid().detach()
            cls_l = F.binary_cross_entropy_with_logits(
                sc, soft, reduction='none').sum(-1)
            cls_total = cls_total + (cls_l * w).sum() / w_sum

            box_l = (sb - tb.detach()).abs().sum(-1)
            box_total = box_total + (box_l * w).sum() / w_sum

            giou_l = self._giou(
                bbox_cxcywh_to_xyxy(sb),
                bbox_cxcywh_to_xyxy(tb.detach()))
            if giou_l.dim() > 1:
                giou_l = giou_l.mean(-1)
            giou_total = giou_total + (giou_l * w).sum() / w_sum

            # --- symmetric half of the supervision ---------------------------
            # Every student query the teacher did not back must be told to stay
            # low, or the loss is a one-way ratchet on scores. The target is the
            # teacher's own score on the query it holds at that location, which
            # is near zero for all of them -- the teacher is as confident that
            # nothing is there as it is that something is somewhere else.
            if cfg['suppress_weight'] > 0:
                mask = torch.ones(s_cls.size(1), dtype=torch.bool,
                                  device=s_cls.device)
                mask[col] = False
                if mask.any():
                    s_un = s_cls[i][mask]
                    t_bg = t_cls[i].sigmoid().detach()
                    # Lowest-scoring teacher queries define "nothing here".
                    bg_target = t_bg.min(dim=0).values.expand(s_un.size(0), -1)
                    sup_l = F.binary_cross_entropy_with_logits(
                        s_un, bg_target, reduction='none').sum(-1)
                    # Weight each unmatched query by the student's OWN
                    # confidence. A plain mean over ~874 queries buries the
                    # handful that actually duplicate a real detection (~34x
                    # dilution, measured at 0.126 total); weighting by the
                    # student's score puts the loss on exactly the offenders --
                    # confident where the teacher backs nothing.
                    sw = s_un.sigmoid().max(-1).values.detach()
                    # Normalised by the number of matched pairs, NOT by the
                    # weight sum: dividing by sw.sum() would make this a
                    # weighted mean, which scores a model with twice the
                    # duplicates the same as one with few (measured: 0.659 vs
                    # 0.643 on exactly that pair of checkpoints). Dividing by
                    # the pair count keeps the penalty growing with the number
                    # of offenders, which is the quantity that costs AP.
                    sup_total = sup_total + (
                        (sup_l * sw).sum() / max(len(col), 1))

            n_pairs += 1

        if n_pairs == 0:
            return zero, zero, zero, zero
        return (cls_total / n_pairs, box_total / n_pairs,
                giou_total / n_pairs, sup_total / n_pairs)

    def _select_teacher(self, t_cls_i):
        """Indices of the teacher detections worth imitating, one image."""
        cfg = self.distill_cfg
        conf = t_cls_i.sigmoid().max(-1).values
        keep = (conf >= cfg['score_thr']).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            return keep
        if keep.numel() > cfg['topk']:
            top = conf[keep].topk(cfg['topk']).indices
            keep = keep[top]
        return keep

    @torch.no_grad()
    def _match(self, s_box_i, t_box_sel):
        """Hungarian-match the selected teacher boxes to student queries.

        Box geometry only: the point is to pair the same physical detection in
        both models, so class scores -- the very thing being distilled -- must
        not steer the pairing.
        """
        cfg = self.distill_cfg
        l1 = torch.cdist(t_box_sel, s_box_i, p=1)

        t_xyxy = bbox_cxcywh_to_xyxy(t_box_sel)
        s_xyxy = bbox_cxcywh_to_xyxy(s_box_i)
        giou = _generalized_iou(t_xyxy, s_xyxy)

        cost = cfg['match_l1_cost'] * l1 + cfg['match_giou_cost'] * (1 - giou)
        cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=1e4)
        try:
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        except ValueError:
            return None
        if len(col) == 0:
            return None
        # linear_sum_assignment returns rows in order, so col lines up with
        # t_box_sel row-for-row only when row is the identity -- make it so.
        order = row.argsort()
        return torch.as_tensor(col[order], device=s_box_i.device,
                               dtype=torch.long)


def _generalized_iou(a, b):
    """GIoU between every box in `a` and every box in `b`. Boxes are xyxy."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter

    lt_c = torch.min(a[:, None, :2], b[None, :, :2])
    rb_c = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    iou = inter / union.clamp(min=1e-6)
    return iou - (area_c - union) / area_c.clamp(min=1e-6)
