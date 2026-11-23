"""Interleaved multi-scale encoder (R3), after Lite-DETR (Li et al., CVPR 2023).

Why this and not RT-DETR's hybrid encoder: the gap this is meant to close is
specifically the stride-4 information. v3 dropped that level and lost 1.06 AP
of box quality (oracle ceiling 83.39 -> 82.33), concentrated in the tiny size
bin (-1.56). RT-DETR never feeds the stride-4 level into its encoder at all, so
it cannot recover that; structurally it makes the same choice as v2. Lite-DETR
keeps the level and makes it cheap, which is the lever this needs.

The observation Lite-DETR builds on -- that roughly 75% of the tokens live at
the highest resolution -- holds almost exactly here: stride 4 carries 63,126 of
84,283 tokens (74.9%) at 1333x755.

The scheme: in most layers only the low-resolution levels are *updated*, while
every level is still *read* (deformable attention samples the full value
sequence, so the cheap queries keep access to stride-4 detail). Every k-th
layer, and always the last, updates everything.

Cost, with 6 layers and `update_high_every=3` (layers 2 and 5 do a full
update): 2 layers over 84,283 queries + 4 layers over 21,157, against 6 layers
over 84,283 -- about 50% of the baseline encoder, with stride-4 retained.

Implementation note: mmcv's `BaseTransformerLayer.forward` hardcodes
`temp_key = temp_value = query` for `self_attn`, so a subset query with a full
value cannot be expressed through it. This encoder therefore drives each
layer's own `attentions` / `norms` / `ffns` directly, in the layer's declared
`operation_order`, rather than calling the layer.
"""
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER_SEQUENCE
from mmdet.models.utils.transformer import DetrTransformerEncoder


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class InterleavedTransformerEncoder(DetrTransformerEncoder):
    """Deformable encoder that updates high-resolution levels only sometimes.

    Args:
        high_res_levels (int): how many leading feature levels count as
            high-resolution. The flattened sequence is ordered finest-first, so
            1 means the stride-4 level alone.
        update_high_every (int): a full update happens every k-th layer. The
            last layer always does one, so the memory handed to the decoder is
            never stale.
        All other arguments are passed to `DetrTransformerEncoder`.
    """

    def __init__(self,
                 *args,
                 high_res_levels=1,
                 update_high_every=3,
                 low_level_only=False,
                 ffn_ratio=8,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert high_res_levels >= 1
        assert update_high_every >= 1
        self.high_res_levels = high_res_levels
        self.update_high_every = update_high_every
        self.low_level_only = low_level_only
        self.ffn_ratio = ffn_ratio
        if low_level_only and ffn_ratio and ffn_ratio > 1:
            self._build_light_ffns()

    def _build_light_ffns(self):
        """Narrow FFNs for the high-resolution update steps.

        Lite-DETR runs the low-level (high-resolution) refresh through an FFN
        whose hidden size is cut by lambda=8. That step carries ~75% of the
        tokens, so the full-width FFN there dominates its cost; the paper
        reports the narrow one loses nothing.
        """
        from mmcv.cnn.bricks.transformer import FFN
        # Only the layers that actually run a low-level refresh get one, keyed
        # by layer index. Building one per layer would park unused parameters
        # in the state_dict that never receive a gradient.
        self.light_ffns = nn.ModuleDict()
        for i in sorted(self._full_update_layers()):
            src = self.layers[i].ffns[0]
            self.light_ffns[str(i)] = FFN(
                embed_dims=src.embed_dims,
                feedforward_channels=max(
                    src.feedforward_channels // self.ffn_ratio,
                    src.embed_dims),
                num_fcs=src.num_fcs,
                ffn_drop=0.0,
                add_identity=True)

    def _full_update_layers(self):
        """Indices of the layers that update every level."""
        last = self.num_layers - 1
        idx = {i for i in range(self.num_layers)
               if (i + 1) % self.update_high_every == 0}
        idx.add(last)
        return idx

    @staticmethod
    def _run_layer(layer, query, value, query_pos, key_padding_mask,
                   reference_points, spatial_shapes, level_start_index,
                   ffn_override=None):
        """One layer, with `query` a subset of `value`.

        Follows the layer's own `operation_order`; `pre_norm` is honoured the
        way `BaseTransformerLayer` does, so a post-norm layer (the default
        here) normalises after each block.
        """
        attn_i = norm_i = ffn_i = 0
        x = query
        for op in layer.operation_order:
            if op == 'self_attn':
                identity = x
                if layer.pre_norm:
                    x = layer.norms[norm_i](x)
                x = layer.attentions[attn_i](
                    x,
                    None,
                    value,
                    identity=identity,
                    query_pos=query_pos,
                    key_padding_mask=key_padding_mask,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index)
                attn_i += 1
            elif op == 'norm':
                if not layer.pre_norm:
                    x = layer.norms[norm_i](x)
                norm_i += 1
            elif op == 'ffn':
                identity = x
                if layer.pre_norm:
                    x = layer.norms[norm_i](x)
                ffn = layer.ffns[ffn_i] if ffn_override is None else ffn_override
                x = ffn(x, identity=identity)
                ffn_i += 1
            else:
                raise ValueError(
                    f'{type(layer).__name__} declares operation {op!r}, which '
                    'this encoder does not know how to drive. Use the stock '
                    'DetrTransformerEncoder for such a layer.')
        return x

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_padding_mask=None,
                spatial_shapes=None,
                reference_points=None,
                level_start_index=None,
                valid_ratios=None,
                **kwargs):
        """`query` is (num_token, bs, dims) -- these layers are not batch_first.

        `spatial_shapes` and `level_start_index` always describe the whole
        sequence, because they index the value, not the query.
        """
        assert spatial_shapes is not None and level_start_index is not None, \
            'InterleavedTransformerEncoder needs the multi-scale layout'
        n_levels = spatial_shapes.size(0)
        if self.high_res_levels >= n_levels:
            # Nothing to hold back; behave exactly like the stock encoder.
            return super().forward(
                query, key, value, query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios, **kwargs)

        # The sequence is finest-level-first, so the split is one slice.
        split = int(level_start_index[self.high_res_levels])
        full_update = self._full_update_layers()

        x = query
        for i, layer in enumerate(self.layers):
            refresh_high = i in full_update
            if refresh_high and self.low_level_only:
                # Lite-DETR's low-level step: only the high-resolution tokens
                # are the query. Updating every token here, as a plain "full"
                # layer would, redoes work the preceding layers just did on the
                # low-resolution ones.
                sl = slice(None, split)
            elif refresh_high:
                sl = slice(None)
            else:
                sl = slice(split, None)

            q = x[sl]
            q_pos = None if query_pos is None else query_pos[sl]
            ref = None if reference_points is None \
                else reference_points[:, sl]
            ffn_override = None
            if refresh_high and self.low_level_only and \
                    hasattr(self, 'light_ffns'):
                key = str(i)
                if key in self.light_ffns:
                    ffn_override = self.light_ffns[key]

            out = self._run_layer(
                layer,
                query=q,
                value=x,
                query_pos=q_pos,
                key_padding_mask=key_padding_mask,
                reference_points=ref,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                ffn_override=ffn_override)

            if sl == slice(None):
                x = out
            elif sl.start == split:
                # High-resolution tokens ride through untouched this layer.
                x = torch.cat([x[:split], out], dim=0)
            else:
                # Low-resolution tokens ride through this one.
                x = torch.cat([out, x[split:]], dim=0)
        return x
