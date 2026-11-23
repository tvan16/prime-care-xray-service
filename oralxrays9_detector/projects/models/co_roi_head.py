import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class CoStandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

            bbox_targets = bbox_results['bbox_targets']
            num_imgs = len(img_metas)
            max_proposal = 2000
            for res in sampling_results:
                max_proposal =  min(max_proposal, res.bboxes.shape[0])
            ori_coords = bbox2roi([res.bboxes for res in sampling_results])
            ori_proposals, ori_labels, ori_bbox_targets, ori_bbox_feats = [], [], [], []
            for i in range(num_imgs):
                idx = (ori_coords[:,0]==i).nonzero().squeeze(1)
                idx = idx[:max_proposal]
                ori_proposal = ori_coords[idx][:, 1:].unsqueeze(0)
                ori_label = bbox_targets[0][idx].unsqueeze(0)
                ori_bbox_target = bbox_targets[2][idx].unsqueeze(0)
                ori_bbox_feat = bbox_results['bbox_feats'].mean(-1).mean(-1)
                ori_bbox_feat = ori_bbox_feat[idx].unsqueeze(0)
                ori_proposals.append(ori_proposal) 
                ori_labels.append(ori_label)
                ori_bbox_targets.append(ori_bbox_target)
                ori_bbox_feats.append(ori_bbox_feat)
            ori_coords = torch.cat(ori_proposals, dim=0)
            ori_labels = torch.cat(ori_labels, dim=0)
            ori_bbox_targets = torch.cat(ori_bbox_targets, dim=0)
            ori_bbox_feats = torch.cat(ori_bbox_feats, dim=0)
            pos_coords = (ori_coords, ori_labels, ori_bbox_targets, ori_bbox_feats, 'rcnn')
            losses.update(pos_coords=pos_coords)

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        bbox_results.update(bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
        return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels


@HEADS.register_module()
class FIRoIHead(CoStandardRoIHead):
    """MOQAM RoI head with CB-SCR regularization.

    The detector still uses the standard Co-DETR RoI auxiliary branch for
    proposal supervision and positive-query mining. This subclass adds the
    two MLPs found in the OralXrays-9 checkpoint (``feature_fc`` and the
    original misspelled ``centure_fc``) and computes a class-balanced
    spherical contrastive regularizer during training.
    """

    def __init__(self,
                 temperature=1.0,
                 num_classes=80,
                 scr_loss_weight=0.5,
                 feature_dim=1024,
                 hidden_dim=512,
                 **kwargs):
        self.temperature = temperature
        self.num_classes = num_classes
        self.scr_loss_weight = scr_loss_weight
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        super().__init__(**kwargs)
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim))
        # Keep the checkpoint spelling exactly: "centure", not "center".
        self.centure_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim))

    def _bbox_embedding(self, bbox_feats):
        """Return the Shared2FC hidden representation used by CB-SCR."""
        bbox_head = self.bbox_head
        x = bbox_feats
        if getattr(bbox_head, 'num_shared_convs', 0) > 0:
            for conv in bbox_head.shared_convs:
                x = conv(x)
        if getattr(bbox_head, 'num_shared_fcs', 0) > 0:
            if bbox_head.with_avg_pool:
                x = bbox_head.avg_pool(x)
            x = x.flatten(1)
            for fc in bbox_head.shared_fcs:
                x = bbox_head.relu(fc(x))
        elif x.dim() > 2:
            x = x.mean(-1).mean(-1)
        return x

    def _bbox_forward(self, x, rois):
        bbox_results = super()._bbox_forward(x, rois)
        bbox_results['bbox_embed'] = self._bbox_embedding(
            bbox_results['bbox_feats'])
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        bbox_results = super()._bbox_forward_train(
            x, sampling_results, gt_bboxes, gt_labels, img_metas)
        labels = bbox_results['bbox_targets'][0]
        loss_cb_scr = self.loss_cb_scr(bbox_results['bbox_embed'], labels)
        bbox_results['loss_bbox']['loss_cb_scr'] = loss_cb_scr
        return bbox_results

    def loss_cb_scr(self, bbox_embed, labels):
        """Class-balanced spherical contrastive regularization."""
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        if pos_inds.numel() < 2:
            return bbox_embed.sum() * 0

        pos_feats = bbox_embed[pos_inds]
        pos_labels = labels[pos_inds]
        present_labels = torch.unique(pos_labels)
        if present_labels.numel() < 2:
            return bbox_embed.sum() * 0

        sample_feats = F.normalize(self.feature_fc(pos_feats), dim=1)
        class_means = []
        for class_id in present_labels:
            class_means.append(pos_feats[pos_labels == class_id].mean(0))
        class_means = torch.stack(class_means, dim=0)
        proto_feats = F.normalize(self.centure_fc(class_means), dim=1)

        bank_feats = torch.cat([sample_feats, proto_feats], dim=0)
        bank_labels = torch.cat([pos_labels, present_labels], dim=0)
        sim = torch.exp(
            torch.mm(sample_feats, bank_feats.t()) /
            max(float(self.temperature), 1e-6))

        losses = []
        eps = sim.new_tensor(1e-6)
        for i in range(sample_feats.size(0)):
            same_class = bank_labels == pos_labels[i]
            same_class[i] = False
            numerator = sim[i][same_class].sum().clamp_min(eps)

            denominator = sim.new_tensor(0.)
            for class_id in present_labels:
                class_mask = bank_labels == class_id
                denominator = denominator + (
                    sim[i][class_mask].sum() /
                    class_mask.float().sum().clamp_min(1.0))
            losses.append(-torch.log(numerator / denominator.clamp_min(eps)))

        return torch.stack(losses).mean() * self.scr_loss_weight


@HEADS.register_module()
class CorrectedFIRoIHead(FIRoIHead):
    """FIRoIHead with a non-negative supervised contrastive objective."""

    def loss_cb_scr(self, bbox_embed, labels):
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        if pos_inds.numel() < 2:
            return bbox_embed.sum() * 0

        pos_feats = bbox_embed[pos_inds]
        pos_labels = labels[pos_inds]
        present_labels = torch.unique(pos_labels)
        if present_labels.numel() < 2:
            return bbox_embed.sum() * 0

        sample_feats = F.normalize(self.feature_fc(pos_feats), dim=1)
        class_means = torch.stack([
            pos_feats[pos_labels == class_id].mean(0)
            for class_id in present_labels
        ])
        proto_feats = F.normalize(self.centure_fc(class_means), dim=1)
        bank_feats = torch.cat([sample_feats, proto_feats], dim=0)
        bank_labels = torch.cat([pos_labels, present_labels], dim=0)
        logits = torch.mm(sample_feats, bank_feats.t()) / max(
            float(self.temperature), 1e-6)

        class_losses = []
        for class_id in present_labels:
            sample_ids = (pos_labels == class_id).nonzero().squeeze(1)
            losses = []
            for sample_id in sample_ids:
                valid = torch.ones(
                    bank_feats.size(0), dtype=torch.bool, device=logits.device)
                valid[sample_id] = False
                positives = (bank_labels == class_id) & valid
                log_prob = logits[sample_id] - torch.logsumexp(
                    logits[sample_id][valid], dim=0)
                losses.append(-log_prob[positives].mean())
            class_losses.append(torch.stack(losses).mean())

        return torch.stack(class_losses).mean() * self.scr_loss_weight


@HEADS.register_module()
class GlobalPrototypeFIRoIHead(FIRoIHead):
    """FIRoIHead with EMA class prototypes for CB-SCR.

    The original CB-SCR computes prototypes from the positive RoIs in the
    current mini-batch. With small batches and class imbalance, many iterations
    contain only a few positive classes. This variant keeps an EMA prototype per
    class so the contrastive denominator can use class evidence accumulated
    across training iterations.
    """

    def __init__(self,
                 prototype_momentum=0.99,
                 global_loss_weight=1.0,
                 local_loss_weight=0.25,
                 min_global_classes=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.prototype_momentum = prototype_momentum
        self.global_loss_weight = global_loss_weight
        self.local_loss_weight = local_loss_weight
        self.min_global_classes = min_global_classes
        self.register_buffer(
            'global_prototypes',
            torch.zeros(self.num_classes, self.feature_dim))
        self.register_buffer(
            'prototype_initialized',
            torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer(
            'prototype_counts',
            torch.zeros(self.num_classes))

    @torch.no_grad()
    def _update_global_prototypes(self, pos_feats, pos_labels):
        for class_id in torch.unique(pos_labels):
            class_int = int(class_id.item())
            class_mask = pos_labels == class_id
            class_mean = pos_feats[class_mask].mean(0)
            if self.prototype_initialized[class_int]:
                momentum = float(self.prototype_momentum)
                self.global_prototypes[class_int].mul_(momentum).add_(
                    class_mean, alpha=1.0 - momentum)
            else:
                self.global_prototypes[class_int].copy_(class_mean)
                self.prototype_initialized[class_int] = True
            self.prototype_counts[class_int] += class_mask.float().sum()

    def _loss_global_cb_scr(self, sample_feats, pos_labels):
        valid_classes = self.prototype_initialized.nonzero().squeeze(1)
        if valid_classes.numel() < self.min_global_classes:
            return sample_feats.sum() * 0

        raw_prototypes = self.global_prototypes[valid_classes]
        proto_feats = F.normalize(self.centure_fc(raw_prototypes), dim=1)
        sim = torch.exp(
            torch.mm(sample_feats, proto_feats.t()) /
            max(float(self.temperature), 1e-6))

        losses = []
        eps = sim.new_tensor(1e-6)
        for i in range(sample_feats.size(0)):
            same_class = valid_classes == pos_labels[i]
            if not same_class.any():
                continue
            numerator = sim[i][same_class].sum().clamp_min(eps)
            denominator = sim[i].sum().clamp_min(eps)
            losses.append(-torch.log(numerator / denominator))

        if not losses:
            return sample_feats.sum() * 0
        return torch.stack(losses).mean()

    def loss_cb_scr(self, bbox_embed, labels):
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        if pos_inds.numel() < 2:
            return bbox_embed.sum() * 0

        pos_feats = bbox_embed[pos_inds]
        pos_labels = labels[pos_inds]
        sample_feats = F.normalize(self.feature_fc(pos_feats), dim=1)
        self._update_global_prototypes(pos_feats.detach(), pos_labels.detach())

        global_loss = self._loss_global_cb_scr(sample_feats, pos_labels)
        if self.local_loss_weight > 0:
            local_loss = super().loss_cb_scr(bbox_embed, labels)
            local_loss = local_loss / max(float(self.scr_loss_weight), 1e-6)
        else:
            local_loss = sample_feats.sum() * 0

        loss = (self.global_loss_weight * global_loss +
                self.local_loss_weight * local_loss)
        return loss * self.scr_loss_weight


@HEADS.register_module()
class ScaleAwareGlobalPrototypeFIRoIHead(GlobalPrototypeFIRoIHead):
    """Global prototype CB-SCR with separate class prototypes per scale bin.

    Dental findings at very different sizes can have different appearances.
    This head keeps EMA prototypes for each (class, scale) pair so small
    lesions are not forced to share the exact same prototype as large findings.
    """

    def __init__(self,
                 num_scale_bins=3,
                 scale_boundaries=(32.0, 96.0),
                 scale_loss_weight=1.0,
                 class_loss_weight=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_scale_bins = num_scale_bins
        self.scale_boundaries = scale_boundaries
        self.scale_loss_weight = scale_loss_weight
        self.class_loss_weight = class_loss_weight
        self.register_buffer(
            'scale_global_prototypes',
            torch.zeros(self.num_classes, num_scale_bins, self.feature_dim))
        self.register_buffer(
            'scale_prototype_initialized',
            torch.zeros(
                self.num_classes, num_scale_bins, dtype=torch.bool))
        self.register_buffer(
            'scale_prototype_counts',
            torch.zeros(self.num_classes, num_scale_bins))

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        labels = bbox_targets[0]
        scale_targets = self._get_scale_targets(sampling_results,
                                                labels.device)
        loss_cb_scr = self.loss_cb_scr(bbox_results['bbox_embed'], labels,
                                       scale_targets)
        loss_bbox['loss_cb_scr'] = loss_cb_scr
        bbox_results.update(loss_bbox=loss_bbox)
        bbox_results.update(bbox_targets=bbox_targets)
        return bbox_results

    def _get_scale_targets(self, sampling_results, device):
        scale_targets = []
        for res in sampling_results:
            targets = torch.full(
                (res.bboxes.size(0), ),
                -1,
                dtype=torch.long,
                device=device)
            num_pos = res.pos_bboxes.size(0)
            if num_pos > 0:
                wh = (res.pos_bboxes[:, 2:] - res.pos_bboxes[:, :2]).clamp_min(
                    0)
                sizes = torch.sqrt(wh[:, 0] * wh[:, 1])
                bins = torch.zeros(num_pos, dtype=torch.long, device=device)
                if self.num_scale_bins >= 2:
                    bins = bins + (sizes >= self.scale_boundaries[0]).long()
                if self.num_scale_bins >= 3:
                    bins = bins + (sizes >= self.scale_boundaries[1]).long()
                bins = bins.clamp(max=self.num_scale_bins - 1)
                targets[:num_pos] = bins
            scale_targets.append(targets)
        return torch.cat(scale_targets)

    @torch.no_grad()
    def _update_scale_global_prototypes(self, pos_feats, pos_labels,
                                        pos_scales):
        keys = torch.unique(
            pos_labels * self.num_scale_bins + pos_scales, sorted=False)
        for key in keys:
            class_int = int((key // self.num_scale_bins).item())
            scale_int = int((key % self.num_scale_bins).item())
            pair_mask = ((pos_labels == class_int) &
                         (pos_scales == scale_int))
            pair_mean = pos_feats[pair_mask].mean(0)
            if self.scale_prototype_initialized[class_int, scale_int]:
                momentum = float(self.prototype_momentum)
                self.scale_global_prototypes[class_int, scale_int].mul_(
                    momentum).add_(pair_mean, alpha=1.0 - momentum)
            else:
                self.scale_global_prototypes[class_int, scale_int].copy_(
                    pair_mean)
                self.scale_prototype_initialized[class_int, scale_int] = True
            self.scale_prototype_counts[class_int,
                                        scale_int] += pair_mask.float().sum()

    def _loss_scale_global_cb_scr(self, sample_feats, pos_labels, pos_scales):
        valid_pairs = self.scale_prototype_initialized.nonzero()
        if valid_pairs.size(0) < self.min_global_classes:
            return sample_feats.sum() * 0

        raw_prototypes = self.scale_global_prototypes[
            valid_pairs[:, 0], valid_pairs[:, 1]]
        proto_feats = F.normalize(self.centure_fc(raw_prototypes), dim=1)
        sim = torch.exp(
            torch.mm(sample_feats, proto_feats.t()) /
            max(float(self.temperature), 1e-6))

        losses = []
        eps = sim.new_tensor(1e-6)
        for i in range(sample_feats.size(0)):
            same_pair = ((valid_pairs[:, 0] == pos_labels[i]) &
                         (valid_pairs[:, 1] == pos_scales[i]))
            if not same_pair.any():
                continue
            numerator = sim[i][same_pair].sum().clamp_min(eps)
            denominator = sim[i].sum().clamp_min(eps)
            losses.append(-torch.log(numerator / denominator))

        if not losses:
            return sample_feats.sum() * 0
        return torch.stack(losses).mean()

    def loss_cb_scr(self, bbox_embed, labels, scale_targets=None):
        if scale_targets is None:
            return super().loss_cb_scr(bbox_embed, labels)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind) &
                    (scale_targets >= 0)).nonzero().squeeze(1)
        if pos_inds.numel() < 2:
            return bbox_embed.sum() * 0

        pos_feats = bbox_embed[pos_inds]
        pos_labels = labels[pos_inds]
        pos_scales = scale_targets[pos_inds]
        sample_feats = F.normalize(self.feature_fc(pos_feats), dim=1)

        detached_feats = pos_feats.detach()
        detached_labels = pos_labels.detach()
        detached_scales = pos_scales.detach()
        self._update_global_prototypes(detached_feats, detached_labels)
        self._update_scale_global_prototypes(detached_feats, detached_labels,
                                             detached_scales)

        class_loss = self._loss_global_cb_scr(sample_feats, pos_labels)
        scale_loss = self._loss_scale_global_cb_scr(sample_feats, pos_labels,
                                                   pos_scales)
        if self.local_loss_weight > 0:
            local_loss = FIRoIHead.loss_cb_scr(self, bbox_embed, labels)
            local_loss = local_loss / max(float(self.scr_loss_weight), 1e-6)
        else:
            local_loss = sample_feats.sum() * 0

        loss = (self.scale_loss_weight * scale_loss +
                self.class_loss_weight * class_loss +
                self.local_loss_weight * local_loss)
        return loss * self.scr_loss_weight


@HEADS.register_module()
class HierarchicalScaleAwareGlobalPrototypeFIRoIHead(
        ScaleAwareGlobalPrototypeFIRoIHead):
    """Scale-aware CB-SCR with stable GT prototypes and hierarchical loss."""

    def __init__(self,
                 normalize_scale=True,
                 prototype_warmup_iters=500,
                 count_aware_momentum=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.normalize_scale = normalize_scale
        self.prototype_warmup_iters = prototype_warmup_iters
        self.count_aware_momentum = count_aware_momentum
        self.register_buffer(
            'prototype_update_steps', torch.zeros((), dtype=torch.long))

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        labels = bbox_targets[0]
        scale_targets = self._get_scale_targets(
            sampling_results, img_metas, labels.device)
        components = self.loss_cb_scr_components(
            bbox_results['bbox_embed'], labels, scale_targets)
        loss_bbox['loss_cb_scr'] = components['total']
        # MMDetection logs these tensors but excludes them from total loss.
        loss_bbox['cb_scr_class'] = components['class'].detach()
        loss_bbox['cb_scr_scale'] = components['scale'].detach()
        loss_bbox['cb_scr_local'] = components['local'].detach()
        bbox_results.update(loss_bbox=loss_bbox)
        bbox_results.update(bbox_targets=bbox_targets)
        return bbox_results

    def _get_scale_targets(self, sampling_results, img_metas, device):
        scale_targets = []
        for res, img_meta in zip(sampling_results, img_metas):
            targets = torch.full(
                (res.bboxes.size(0), ),
                -1,
                dtype=torch.long,
                device=device)
            num_pos = res.pos_gt_bboxes.size(0)
            if num_pos > 0:
                wh = (res.pos_gt_bboxes[:, 2:] -
                      res.pos_gt_bboxes[:, :2]).clamp_min(0)
                sizes = torch.sqrt(wh[:, 0] * wh[:, 1])
                if self.normalize_scale:
                    img_h, img_w = img_meta['img_shape'][:2]
                    image_scale = sizes.new_tensor(
                        float(img_h * img_w)).sqrt().clamp_min(1.0)
                    sizes = sizes / image_scale

                bins = torch.zeros(num_pos, dtype=torch.long, device=device)
                if self.num_scale_bins >= 2:
                    bins += (sizes >= self.scale_boundaries[0]).long()
                if self.num_scale_bins >= 3:
                    bins += (sizes >= self.scale_boundaries[1]).long()
                targets[:num_pos] = bins.clamp(max=self.num_scale_bins - 1)
            scale_targets.append(targets)
        return torch.cat(scale_targets)

    def _adaptive_momentum(self, old_count, batch_count):
        if not self.count_aware_momentum:
            return float(self.prototype_momentum)
        cumulative_momentum = float(
            old_count / (old_count + batch_count).clamp_min(1.0))
        return min(float(self.prototype_momentum), cumulative_momentum)

    @torch.no_grad()
    def _update_global_prototypes(self, pos_feats, pos_labels):
        for class_id in torch.unique(pos_labels):
            class_int = int(class_id.item())
            class_mask = pos_labels == class_id
            batch_count = class_mask.float().sum()
            class_mean = pos_feats[class_mask].mean(0)
            if self.prototype_initialized[class_int]:
                momentum = self._adaptive_momentum(
                    self.prototype_counts[class_int], batch_count)
                self.global_prototypes[class_int].mul_(momentum).add_(
                    class_mean, alpha=1.0 - momentum)
            else:
                self.global_prototypes[class_int].copy_(class_mean)
                self.prototype_initialized[class_int] = True
            self.prototype_counts[class_int] += batch_count

    @torch.no_grad()
    def _update_scale_global_prototypes(self, pos_feats, pos_labels,
                                        pos_scales):
        keys = torch.unique(
            pos_labels * self.num_scale_bins + pos_scales, sorted=False)
        for key in keys:
            class_int = int((key // self.num_scale_bins).item())
            scale_int = int((key % self.num_scale_bins).item())
            pair_mask = ((pos_labels == class_int) &
                         (pos_scales == scale_int))
            batch_count = pair_mask.float().sum()
            pair_mean = pos_feats[pair_mask].mean(0)
            if self.scale_prototype_initialized[class_int, scale_int]:
                momentum = self._adaptive_momentum(
                    self.scale_prototype_counts[class_int, scale_int],
                    batch_count)
                self.scale_global_prototypes[class_int, scale_int].mul_(
                    momentum).add_(pair_mean, alpha=1.0 - momentum)
            else:
                self.scale_global_prototypes[class_int, scale_int].copy_(
                    pair_mean)
                self.scale_prototype_initialized[class_int, scale_int] = True
            self.scale_prototype_counts[class_int, scale_int] += batch_count

    def _loss_scale_global_cb_scr(self, sample_feats, pos_labels, pos_scales):
        valid_pairs = self.scale_prototype_initialized.nonzero()
        if valid_pairs.size(0) < self.min_global_classes:
            return sample_feats.sum() * 0

        raw_prototypes = self.scale_global_prototypes[
            valid_pairs[:, 0], valid_pairs[:, 1]]
        proto_feats = F.normalize(self.centure_fc(raw_prototypes), dim=1)
        sim = torch.exp(
            torch.mm(sample_feats, proto_feats.t()) /
            max(float(self.temperature), 1e-6))

        losses = []
        eps = sim.new_tensor(1e-6)
        for i in range(sample_feats.size(0)):
            same_class = valid_pairs[:, 0] == pos_labels[i]
            same_pair = same_class & (valid_pairs[:, 1] == pos_scales[i])
            different_class = ~same_class
            if not same_pair.any() or not different_class.any():
                continue
            numerator = sim[i][same_pair].sum().clamp_min(eps)
            denominator = (
                numerator + sim[i][different_class].sum()).clamp_min(eps)
            losses.append(-torch.log(numerator / denominator))

        if not losses:
            return sample_feats.sum() * 0
        return torch.stack(losses).mean()

    def loss_cb_scr_components(self, bbox_embed, labels, scale_targets):
        zero = bbox_embed.sum() * 0
        components = {
            'total': zero,
            'class': zero,
            'scale': zero,
            'local': zero
        }
        pos_inds = ((labels >= 0) & (labels < self.num_classes) &
                    (scale_targets >= 0)).nonzero().squeeze(1)
        if pos_inds.numel() == 0:
            return components

        pos_feats = bbox_embed[pos_inds]
        pos_labels = labels[pos_inds]
        pos_scales = scale_targets[pos_inds]

        # BatchNorm in feature_fc needs at least two positive RoIs. Sparse
        # dental images can yield a single positive, especially at batch size 1;
        # still use it to update the prototype bank but skip contrastive loss.
        if pos_inds.numel() < 2:
            self._update_global_prototypes(
                pos_feats.detach(), pos_labels.detach())
            self._update_scale_global_prototypes(
                pos_feats.detach(), pos_labels.detach(), pos_scales.detach())
            self.prototype_update_steps += 1
            return components

        sample_feats = F.normalize(self.feature_fc(pos_feats), dim=1)

        ready = int(self.prototype_update_steps.item()) >= int(
            self.prototype_warmup_iters)
        if ready:
            class_loss = self._loss_global_cb_scr(sample_feats, pos_labels)
            scale_loss = self._loss_scale_global_cb_scr(
                sample_feats, pos_labels, pos_scales)
        else:
            class_loss = zero
            scale_loss = zero

        if self.local_loss_weight > 0 and pos_inds.numel() >= 2:
            local_loss = FIRoIHead.loss_cb_scr(self, bbox_embed, labels)
            local_loss = local_loss / max(float(self.scr_loss_weight), 1e-6)
        else:
            local_loss = zero

        components['class'] = (
            self.scr_loss_weight * self.class_loss_weight * class_loss)
        components['scale'] = (
            self.scr_loss_weight * self.scale_loss_weight * scale_loss)
        components['local'] = (
            self.scr_loss_weight * self.local_loss_weight * local_loss)
        components['total'] = (components['class'] + components['scale'] +
                               components['local'])

        self._update_global_prototypes(pos_feats.detach(), pos_labels.detach())
        self._update_scale_global_prototypes(
            pos_feats.detach(), pos_labels.detach(), pos_scales.detach())
        self.prototype_update_steps += 1
        return components

    def loss_cb_scr(self, bbox_embed, labels, scale_targets=None):
        if scale_targets is None:
            return super().loss_cb_scr(bbox_embed, labels)
        return self.loss_cb_scr_components(
            bbox_embed, labels, scale_targets)['total']
