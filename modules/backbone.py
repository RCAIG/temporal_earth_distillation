# TED backbone: temporal encoder with sequence-state and patch-state heads
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.transformer_blocks import AttentionBlock
from layers.embedding import PositionalEncoding
from utils.tools import patchify, imputator_sliding_window_overlap

from .heads import DINOHead


def lon_lat_fourier_features(lon_lat_deg: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """
    Map discrete lon/lat (degrees, WGS84) to multi-frequency sin/cos for roughly continuous spherical position embedding.
    lon_lat_deg: [B, 2]，column 0=longitude, column 1=latitude.
    Returns [B, 4*n_freqs] float; caller casts to input device/dtype at inference.
    """
    lon = lon_lat_deg[:, 0:1] * (math.pi / 180.0)
    lat = lon_lat_deg[:, 1:2] * (math.pi / 180.0)
    parts = []
    for i in range(n_freqs):
        w = 2.0 ** i
        parts.extend(
            [
                torch.sin(w * lon),
                torch.cos(w * lon),
                torch.sin(w * lat),
                torch.cos(w * lat),
            ]
        )
    return torch.cat(parts, dim=-1)


class Backbone(nn.Module):
    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.n_heads = configs.n_heads
        self.dropout = float(getattr(configs, "dropout", 0.0))
        self.drop_path = float(
            getattr(configs, "drop_path", getattr(configs, "drop_depth", 0.1))
        )
        self.missing_mask_embed_dropout = float(
            getattr(configs, "missing_mask_embed_dropout", 0.0)
        )
        self.use_missing_mask_embed = bool(getattr(configs, "use_missing_mask_embed", True))
        self.e_layers = configs.e_layers
        self.num_patches = math.ceil((self.seq_len - self.patch_len + self.stride) / self.stride)

        self.n_storage_tokens = getattr(configs, 'n_storage_tokens', 2)
        self.n_cls_tokens = max(1, int(getattr(configs, 'n_cls_tokens', 1)))
        self.imputator_segment_stride = getattr(configs, 'imputator_segment_stride', 244)

        patch_dim = self.patch_len * self.enc_in
        self.embedding = nn.Linear(patch_dim, self.d_model)

        self.time_mark_embedding = nn.Linear(self.patch_len * 2, self.d_model)
        self.missing_mask_embedding = nn.Linear(self.patch_len * 1, self.d_model)

        self.use_lon_lat_embed = bool(getattr(configs, "use_lon_lat_embed", True))
        self.lon_lat_n_fourier_freqs = int(getattr(configs, "lon_lat_n_fourier_freqs", 4))
        _fourier_dim = 4 * self.lon_lat_n_fourier_freqs
        if self.use_lon_lat_embed:
            self.lon_lat_proj = nn.Linear(_fourier_dim, self.d_model)
        else:
            self.lon_lat_proj = None  # not a Parameter; old checkpoints may omit this key

        self.position_encoding = PositionalEncoding(self.d_model, self.num_patches)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.cls_token = nn.Parameter(torch.zeros(1, self.n_cls_tokens, self.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        use_cls_gate = not bool(int(getattr(configs, "evidence_gap_distill", 0)))
        if self.n_cls_tokens > 1 and use_cls_gate:
            self.cls_attn_output_gate = nn.Linear(self.d_model, 1, bias=True)
        else:
            self.cls_attn_output_gate = None

        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, self.d_model))
            nn.init.normal_(self.storage_tokens, std=0.02)

        _attn_ckpt = not getattr(configs, 'no_attn_checkpoint', False)
        self.encoder = nn.ModuleList([
            AttentionBlock(
                self.d_model,
                self.n_heads,
                configs.d_ff // self.d_model,
                self.dropout,
                drop_path=self.drop_path,
                use_checkpoint=_attn_ckpt,
            )
            for l in range(self.e_layers)
        ])

        self.norm_student = nn.LayerNorm(self.d_model)
        self.norm_teacher = nn.LayerNorm(self.d_model)

        dino_head_n_prototypes = getattr(configs, 'dino_head_n_prototypes', 512)
        dino_head_hidden_dim = getattr(configs, 'dino_head_hidden_dim', 128)
        dino_head_bottleneck_dim = getattr(configs, 'dino_head_bottleneck_dim', 64)
        dino_head_nlayers = getattr(configs, 'dino_head_nlayers', 1)
        self.dino_head = DINOHead(
            in_dim=self.d_model,
            out_dim=dino_head_n_prototypes,
            hidden_dim=dino_head_hidden_dim,
            bottleneck_dim=dino_head_bottleneck_dim,
            nlayers=dino_head_nlayers,
        )

        ibot_head_n_prototypes = getattr(configs, 'ibot_head_n_prototypes', 512)
        ibot_head_hidden_dim = getattr(configs, 'ibot_head_hidden_dim', 128)
        ibot_head_bottleneck_dim = getattr(configs, 'ibot_head_bottleneck_dim', 64)
        ibot_head_nlayers = getattr(configs, 'ibot_head_nlayers', 1)
        self.ibot_head = DINOHead(
            in_dim=self.d_model,
            out_dim=ibot_head_n_prototypes,
            hidden_dim=ibot_head_hidden_dim,
            bottleneck_dim=ibot_head_bottleneck_dim,
            nlayers=ibot_head_nlayers,
        )

        self.proj_head = self.dino_head

    def _fuse_cls_tokens(self, z_seq):
        z_cls_bank = z_seq[:, : self.n_cls_tokens]
        if self.n_cls_tokens == 1:
            return z_cls_bank[:, 0], z_cls_bank
        if self.cls_attn_output_gate is None:
            return z_cls_bank.mean(dim=1), z_cls_bank
        g = torch.sigmoid(self.cls_attn_output_gate(z_cls_bank))
        z_cls = (z_cls_bank * g).sum(dim=1)
        return z_cls, z_cls_bank

    def _lon_lat_patch_embed(
        self,
        lon_lat: torch.Tensor,
        B: int,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Per sample: lon/lat at first time step (constant pixel location) -> Fourier features -> d_model, broadcast to N patches."""
        if self.lon_lat_proj is None or lon_lat is None:
            return torch.zeros(B, N, self.d_model, device=device, dtype=dtype)
        rep = torch.nan_to_num(lon_lat[:, 0, :].float(), nan=0.0)
        feat = lon_lat_fourier_features(rep, self.lon_lat_n_fourier_freqs).to(dtype=dtype)
        ll = self.lon_lat_proj(feat)
        return ll.unsqueeze(1).expand(B, N, -1)

    def _maybe_drop_missing_mask_embed(
        self,
        missing_mask_embed: torch.Tensor,
        keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training: randomly drop missing embedding per sample; inference always keeps it (infer missing from spectrum/values).

        If keep ([B,1,1]) is passed, apply even in eval (e.g. teacher forward),
        so student can share the same dropout mask.
        """
        if self.missing_mask_embed_dropout <= 0.0 or missing_mask_embed is None:
            return missing_mask_embed
        if keep is None:
            if not self.training:
                return missing_mask_embed
            B = int(missing_mask_embed.shape[0])
            device = missing_mask_embed.device
            keep = (
                torch.rand(B, 1, 1, device=device, dtype=torch.float32)
                >= self.missing_mask_embed_dropout
            ).to(dtype=missing_mask_embed.dtype)
        else:
            keep = keep.to(device=missing_mask_embed.device, dtype=missing_mask_embed.dtype)
        return missing_mask_embed * keep

    def forward(
        self,
        x_enc,
        missing_mask,
        time_mark,
        mask_map=None,
        is_student=True,
        lon_lat=None,
        geo_keep=None,
        missing_mask_embed_keep=None,
        context_target_attn: str = "full",
    ):
        x_patches = patchify(x_enc, self.patch_len, self.stride)
        x_embed = self.embedding(x_patches)
        B, N, D = x_embed.shape

        if time_mark is not None:
            time_mark_patches = patchify(time_mark, self.patch_len, self.stride)
            time_mark_embed = self.time_mark_embedding(time_mark_patches)
        else:
            time_mark_embed = torch.zeros(B, N, D, device=x_enc.device, dtype=x_embed.dtype)

        if self.use_missing_mask_embed:
            missing_mask_patches = patchify(
                missing_mask.unsqueeze(-1).float(), self.patch_len, self.stride
            )
            missing_mask_embed = self.missing_mask_embedding(missing_mask_patches)
            missing_mask_embed = self._maybe_drop_missing_mask_embed(
                missing_mask_embed, keep=missing_mask_embed_keep
            )
        else:
            missing_mask_embed = torch.zeros(B, N, D, device=x_enc.device, dtype=x_embed.dtype)

        ll_emb = self._lon_lat_patch_embed(lon_lat, B, N, x_embed.dtype, x_enc.device)
        if geo_keep is not None:
            ll_emb = ll_emb * geo_keep.to(dtype=ll_emb.dtype)

        x_input = x_embed + time_mark_embed + missing_mask_embed + ll_emb

        if mask_map is not None:
            mask_expand = mask_map.unsqueeze(-1).expand(-1, -1, D).type_as(x_embed)
            mask_tokens = self.mask_token.expand(B, N, -1)
            x_input = x_input * (1 - mask_expand) + mask_tokens * mask_expand

        pos_embed = self.position_encoding(x_input)
        if pos_embed.shape[1] > N:
            pos_embed = pos_embed[:, :N, :]
        elif pos_embed.shape[1] < N:
            pad_len = N - pos_embed.shape[1]
            pos_embed = torch.cat([pos_embed, torch.zeros(B, pad_len, D, device=pos_embed.device, dtype=pos_embed.dtype)], dim=1)
        x_input = x_input + pos_embed

        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens.expand(B, -1, -1)
            x_input = torch.cat([cls_tokens, storage_tokens, x_input], dim=1)
        else:
            x_input = torch.cat([cls_tokens, x_input], dim=1)

        # Optional JEPA-CE isolation: only when explicitly requested + mask_map present.
        # Default context_target_attn="full" keeps historical bidirectional attention.
        layer_attn_mask = None
        if (
            mask_map is not None
            and str(context_target_attn).lower() == "disjoint"
        ):
            from utils.jepa_masking import build_jepa_disjoint_attn_mask

            n_prefix = int(self.n_cls_tokens) + int(self.n_storage_tokens)
            layer_attn_mask = build_jepa_disjoint_attn_mask(
                mask_map,
                n_prefix=n_prefix,
                num_heads=int(self.n_heads),
                dtype=x_input.dtype,
            )

        for layer in self.encoder:
            x_input, _ = layer(
                x_input,
                is_causal=False,
                return_attn=False,
                attn_mask=layer_attn_mask,
            )

        if is_student:
            z = self.norm_student(x_input)
        else:
            z = self.norm_teacher(x_input)

        z_cls, z_cls_bank = self._fuse_cls_tokens(z)
        off = self.n_cls_tokens
        if self.n_storage_tokens > 0:
            z_storage = z[:, off : off + self.n_storage_tokens]
            z_patch = z[:, off + self.n_storage_tokens :]
        else:
            z_storage = None
            z_patch = z[:, off:]

        if is_student:
            logits_global = self.dino_head(z_cls)
            logits_cls_bank = self.dino_head(z_cls_bank)
            logits_patch = None
        else:
            with torch.no_grad():
                logits_global = self.dino_head(z_cls)
                logits_cls_bank = self.dino_head(z_cls_bank)
                logits_patch = None

        return {
            'logits_global': logits_global,
            'logits_cls_bank': logits_cls_bank,
            'logits_patch': logits_patch,
            'rec_patches': None,
            'z_cls': z_cls,
            'z_cls_bank': z_cls_bank,
            'z_storage': z_storage,
            'z_global': z_cls,
            'z_patch_enc': z_patch,
        }

    def encode(
        self,
        xEnc,
        missing_mask,
        time_mark,
        imputator=None,
        use_student_norm=False,
        output_attentions=False,
        lon_lat=None,
        geo_keep=None,
        missing_mask_embed_keep=None,
    ):
        B = xEnc.shape[0]
        device = xEnc.device

        if imputator is not None:
            missing_mask_orig = missing_mask.clone()
            x_clean_filled = xEnc.nan_to_num(0.0)
            max_imp_len = getattr(imputator, 'pred_len', 366)
            stride_imp = getattr(self, 'imputator_segment_stride', 244)

            with torch.no_grad():
                imp_device = next(imputator.parameters()).device
                imputed_out = imputator_sliding_window_overlap(
                    xEnc,
                    time_mark,
                    missing_mask_orig.bool(),
                    imputator,
                    window_len=max_imp_len,
                    stride=stride_imp,
                    device=device,
                    imp_device=imp_device,
                )

            mask_expanded = missing_mask_orig.unsqueeze(-1).float()
            xEnc = x_clean_filled * (1 - mask_expanded) + imputed_out * mask_expanded
            missing_mask = torch.zeros_like(missing_mask_orig)

        x_patches = patchify(xEnc, self.patch_len, self.stride)
        x_embed = self.embedding(x_patches)
        N = x_embed.shape[1]

        if time_mark is not None:
            time_mark_patches = patchify(time_mark, self.patch_len, self.stride)
            time_mark_embed = self.time_mark_embedding(time_mark_patches)
        else:
            time_mark_embed = torch.zeros(B, N, self.d_model, device=xEnc.device, dtype=x_embed.dtype)

        if self.use_missing_mask_embed:
            missing_mask_patches = patchify(
                missing_mask.unsqueeze(-1).float(), self.patch_len, self.stride
            )
            missing_mask_embed = self.missing_mask_embedding(missing_mask_patches)
            missing_mask_embed = self._maybe_drop_missing_mask_embed(
                missing_mask_embed, keep=missing_mask_embed_keep
            )
        else:
            missing_mask_embed = torch.zeros(
                B, N, self.d_model, device=xEnc.device, dtype=x_embed.dtype
            )

        ll_emb = self._lon_lat_patch_embed(lon_lat, B, N, x_embed.dtype, xEnc.device)
        if geo_keep is not None:
            ll_emb = ll_emb * geo_keep.to(dtype=ll_emb.dtype)

        x_input_embed = x_embed + time_mark_embed + missing_mask_embed + ll_emb

        pos_embed = self.position_encoding(x_input_embed)
        if pos_embed.shape[1] > N:
            pos_embed = pos_embed[:, :N, :]
        elif pos_embed.shape[1] < N:
            pad_len = N - pos_embed.shape[1]
            pos_embed = torch.cat([pos_embed, torch.zeros(B, pad_len, self.d_model, device=pos_embed.device, dtype=pos_embed.dtype)], dim=1)
        x_input = x_input_embed + pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.n_storage_tokens > 0:
            storage_tokens_in = self.storage_tokens.expand(B, -1, -1)
            x_input = torch.cat([cls_tokens, storage_tokens_in, x_input], dim=1)
        else:
            x_input = torch.cat([cls_tokens, x_input], dim=1)

        all_attns = []
        if output_attentions:
            for layer in self.encoder:
                x_input, attn = layer(x_input, is_causal=False, return_attn=True)
                all_attns.append(attn if attn is not None else None)
        else:
            for layer in self.encoder:
                x_input, _ = layer(x_input, is_causal=False, return_attn=False)

        if use_student_norm:
            x_out = self.norm_student(x_input)
        else:
            x_out = self.norm_teacher(x_input)

        cls_token_fused, cls_token_bank = self._fuse_cls_tokens(x_out)
        off = self.n_cls_tokens
        if self.n_storage_tokens > 0:
            storage_tokens = x_out[:, off : off + self.n_storage_tokens]
            patch_tokens = x_out[:, off + self.n_storage_tokens :]
        else:
            storage_tokens = None
            patch_tokens = x_out[:, off:]

        result = {
            'cls_token': cls_token_fused,
            'cls_token_bank': cls_token_bank,
            'storage_tokens': storage_tokens,
            'patch_tokens': patch_tokens,
        }

        if output_attentions:
            z_cls_expanded = cls_token_fused.unsqueeze(1)
            cls_cos_sim_all = F.cosine_similarity(z_cls_expanded, x_out, dim=-1)
            cls_cos_sim_patch = F.cosine_similarity(z_cls_expanded, patch_tokens, dim=-1)

            result.update({
                'all_attns': all_attns,
                'cls_cos_sim_all': cls_cos_sim_all,
                'cls_cos_sim_patch': cls_cos_sim_patch,
            })

        return result

    def visualize(
        self,
        xEnc,
        missing_mask,
        time_mark,
        imputator=None,
        use_student_norm=False,
        lon_lat=None,
        geo_keep=None,
    ):
        return self.encode(
            xEnc,
            missing_mask,
            time_mark,
            imputator=imputator,
            use_student_norm=use_student_norm,
            output_attentions=True,
            lon_lat=lon_lat,
            geo_keep=geo_keep,
        )
