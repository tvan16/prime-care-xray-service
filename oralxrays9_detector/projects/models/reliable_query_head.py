import torch
import torch.distributed as dist
import torch.nn.functional as F

from mmdet.models.builder import HEADS
from .co_dino_head import QualityAwareHierarchicalCoDINOHead


@HEADS.register_module()
class ReliableQueryCoDINOHead(QualityAwareHierarchicalCoDINOHead):
    """Single-GPU query memory with quality-weighted, decayed moments.

    R2 uses cosine logits; R3 uses diagonal Gaussian energy. Reliability is
    an experimental heuristic, not a calibrated probability of correctness.
    """

    def __init__(self, prototype_distance='cosine', support_tau=32.0,
                 variance_prior=0.004, variance_floor=0.001,
                 variance_prior_mass=8.0, dispersion_tau=0.004,
                 **kwargs):
        super().__init__(**kwargs)
        if prototype_distance not in ('cosine', 'gaussian'):
            raise ValueError('prototype_distance must be cosine or gaussian')
        if not 0 < self.prototype_momentum < 1:
            raise ValueError('prototype_momentum must be between zero and one')
        if min(support_tau, variance_floor, variance_prior_mass,
               dispersion_tau, self.query_proto_temperature) <= 0:
            raise ValueError('Reliability and distance constants must be positive')
        if variance_prior < variance_floor:
            raise ValueError('variance_prior must be >= variance_floor')
        if not 0 <= self.scale_interpolation <= 1:
            raise ValueError('scale_interpolation must be between zero and one')
        self.prototype_distance = prototype_distance
        self.support_tau = support_tau
        self.variance_prior = variance_prior
        self.variance_floor = variance_floor
        self.variance_prior_mass = variance_prior_mass
        self.dispersion_tau = dispersion_tau
        bins = len(self.scale_boundaries) + 1
        # Index zero pools all scales. Remaining indices hold individual bins.
        shape = (self.num_classes, bins + 1)
        self.register_buffer('reliable_mass', torch.zeros(*shape))
        self.register_buffer('reliable_sum', torch.zeros(*shape, self.embed_dims))
        self.register_buffer('reliable_square_sum',
                             torch.zeros(*shape, self.embed_dims))

    @torch.no_grad()
    def _update_reliable_memory(self, feats, labels, bins, weights):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError('Reliable query experiments currently require one GPU')
        # All groups decay, including absent classes, so stale evidence loses weight.
        for value in (self.reliable_mass, self.reliable_sum,
                      self.reliable_square_sum):
            value.mul_(self.prototype_momentum)
        for class_id in labels.unique():
            c = int(class_id)
            for slot in range(self.reliable_mass.size(1)):
                mask = labels == c
                if slot:
                    mask = mask & (bins == slot - 1)
                x, w = feats[mask].float(), weights[mask].detach().float()
                self.reliable_mass[c, slot].add_(w.sum())
                self.reliable_sum[c, slot].add_((x * w[:, None]).sum(0))
                self.reliable_square_sum[c, slot].add_(
                    (x.square() * w[:, None]).sum(0))
        self.query_proto_steps.add_(1)

    def _memory_snapshot(self):
        # Clone before updating memory: backward needs the previous batch's values.
        mass = self.reliable_mass.detach().clone()
        mean = self.reliable_sum.detach().clone() / mass.clamp_min(1e-6)[..., None]
        second = self.reliable_square_sum.detach().clone()
        raw_var = (second / mass.clamp_min(1e-6)[..., None] - mean.square()).clamp_min(0)
        prior_weight = mass / (mass + self.variance_prior_mass)
        variance = (prior_weight[..., None] * raw_var +
                    (1 - prior_weight[..., None]) * self.variance_prior)
        variance = variance.clamp_min(self.variance_floor)
        support = mass / (mass + self.support_tau)
        reliability = support / (1 + variance.mean(-1) / self.dispersion_tau)
        return mass, mean, variance, reliability

    def _reliable_prototype_loss(self, feats, labels, bins, weights):
        zero = feats.sum() * 0
        if self.query_proto_steps.item() < self.prototype_warmup_iters:
            return zero, zero.detach(), zero.detach()
        mass, mean, variance, reliability = self._memory_snapshot()
        valid = mass[:, 0] > 1e-4
        if valid.sum() < 2:
            return zero, zero.detach(), zero.detach()
        losses, loss_weights, anchor_reliabilities, interpolations = [], [], [], []
        for feat, label, bin_id, quality in zip(feats, labels, bins, weights):
            c, slot = int(label), int(bin_id) + 1
            if not valid[c]:
                continue
            # Apply the same class/scale construction to positives and negatives.
            alpha = self.scale_interpolation * reliability[:, slot]
            a = alpha[:, None]
            centers = (1 - a) * mean[:, 0] + a * mean[:, slot]
            mixed_var = ((1 - a) * variance[:, 0] + a * variance[:, slot] +
                         a * (1 - a) * (mean[:, 0] - mean[:, slot]).square())
            centers = centers[valid]
            if self.prototype_distance == 'cosine':
                logits = F.normalize(centers, dim=-1) @ feat.float()
            else:
                v = mixed_var[valid].clamp_min(self.variance_floor)
                logits = -0.5 * (((feat.float() - centers).square() / v) +
                                 v.log()).mean(-1)
            logits = logits / self.query_proto_temperature
            target = valid[:c].sum().reshape(1).long()
            losses.append(F.cross_entropy(logits[None], target))
            r = reliability[c, 0]
            loss_weights.append(quality.detach() * r)
            anchor_reliabilities.append(r)
            interpolations.append(alpha[c])
        if not losses:
            return zero, zero.detach(), zero.detach()
        # Normalize by number of anchors so uncertain memory reduces the force.
        loss = (torch.stack(losses) * torch.stack(loss_weights)).mean()
        return (loss, torch.stack(anchor_reliabilities).mean(),
                torch.stack(interpolations).mean())

    def _prototype_regularization(self, feats, labels, scale_bins, weights):
        loss, reliability, alpha = self._reliable_prototype_loss(
            feats, labels, scale_bins, weights)
        self._reliable_metrics = dict(
            query_memory_reliability=reliability.detach(),
            query_memory_scale_mix=alpha.detach())
        self._update_reliable_memory(feats.detach(), labels, scale_bins, weights)
        return loss

    def _query_regularization(self, *args, **kwargs):
        self._reliable_metrics = {}
        result = super()._query_regularization(*args, **kwargs)
        if not self._reliable_metrics:
            # Empty-positive batches must still age the memory and advance warm-up.
            x = self.reliable_sum.new_empty((0, self.embed_dims))
            labels = torch.empty(0, dtype=torch.long, device=x.device)
            self._update_reliable_memory(x, labels, labels, x.new_empty(0))
            self._reliable_metrics = dict(
                query_memory_reliability=x.new_zeros(()),
                query_memory_scale_mix=x.new_zeros(()))
        result.update(self._reliable_metrics)
        return result
