"""Section 7.3 消融用 forward（不修改 cs336_basics/linear.py 的 baseline 实现）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from cs336_basics.linear import (
    multihead_self_attention,
    multihead_self_attention_with_rope,
    rmsnorm,
    silu,
    swiglu,
)

AblationVariant = Literal["baseline", "no_rmsnorm", "post_norm", "no_rope", "silu_ffn"]
NormStyle = Literal["pre", "post"]
FfnType = Literal["swiglu", "silu"]


@dataclass(frozen=True)
class AblationFlags:
    variant: AblationVariant = "baseline"
    use_rmsnorm: bool = True
    norm_style: NormStyle = "pre"
    use_rope: bool = True
    ffn_type: FfnType = "swiglu"

    @staticmethod
    def from_variant(variant: AblationVariant) -> AblationFlags:
        if variant == "baseline":
            return AblationFlags()
        if variant == "no_rmsnorm":
            return AblationFlags(variant=variant, use_rmsnorm=False)
        if variant == "post_norm":
            return AblationFlags(variant=variant, norm_style="post")
        if variant == "no_rope":
            return AblationFlags(variant=variant, use_rope=False)
        if variant == "silu_ffn":
            return AblationFlags(variant=variant, ffn_type="silu")
        raise ValueError(variant)


def default_d_ff(d_model: int, ffn_type: FfnType) -> int:
    if ffn_type == "swiglu":
        return (d_model * 8) // 3
    return 4 * d_model


def _apply_rmsnorm(d_model: int, weight: Tensor, x: Tensor, *, enabled: bool) -> Tensor:
    if not enabled:
        return x
    return rmsnorm(d_model, 1e-5, weight, x)


def _run_ffn(
    d_model: int,
    d_ff: int,
    ffn_type: FfnType,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor | None,
    x: Tensor,
) -> Tensor:
    if ffn_type == "swiglu":
        assert w3 is not None
        return swiglu(d_model, d_ff, w1, w2, w3, x)
    return (silu(x @ w1.T)) @ w2.T


def _run_attn(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    use_rope: bool,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    x: Tensor,
    batch_size: int,
    seq_len: int,
) -> Tensor:
    if use_rope:
        token_positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        return multihead_self_attention_with_rope(
            d_model, num_heads, max_seq_len, theta, q, k, v, o, x, token_positions
        )
    return multihead_self_attention(d_model, num_heads, q, k, v, o, x)


def transformer_block_ablation(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    flags: AblationFlags,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    batch_size, seq_len, _ = in_features.shape
    w3 = weights.get("ffn.w3.weight")

    if flags.norm_style == "pre":
        h = _apply_rmsnorm(d_model, weights["ln1.weight"], in_features, enabled=flags.use_rmsnorm)
        attn = _run_attn(
            d_model,
            num_heads,
            max_seq_len,
            theta,
            flags.use_rope,
            weights["attn.q_proj.weight"],
            weights["attn.k_proj.weight"],
            weights["attn.v_proj.weight"],
            weights["attn.output_proj.weight"],
            h,
            batch_size,
            seq_len,
        )
        z = in_features + attn
        h2 = _apply_rmsnorm(d_model, weights["ln2.weight"], z, enabled=flags.use_rmsnorm)
        ffn = _run_ffn(
            d_model,
            d_ff,
            flags.ffn_type,
            weights["ffn.w1.weight"],
            weights["ffn.w2.weight"],
            w3,
            h2,
        )
        return z + ffn

    attn = _run_attn(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        flags.use_rope,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        in_features,
        batch_size,
        seq_len,
    )
    z = _apply_rmsnorm(d_model, weights["ln1.weight"], in_features + attn, enabled=flags.use_rmsnorm)
    ffn = _run_ffn(
        d_model,
        d_ff,
        flags.ffn_type,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        w3,
        z,
    )
    return _apply_rmsnorm(d_model, weights["ln2.weight"], z + ffn, enabled=flags.use_rmsnorm)


def transformer_lm_ablation(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    flags: AblationFlags,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    x = weights["token_embeddings.weight"][in_indices]
    for i in range(num_layers):
        prefix = f"layers.{i}."
        block_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        x = transformer_block_ablation(
            d_model,
            num_heads,
            d_ff,
            context_length,
            rope_theta,
            flags,
            block_weights,
            x,
        )
    x = _apply_rmsnorm(d_model, weights["ln_final.weight"], x, enabled=flags.use_rmsnorm)
    return x @ weights["lm_head.weight"].T


class AblationTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        flags: AblationFlags,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.flags = flags
        self._key_to_name: dict[str, str] = {}

        def add(key: str, shape: tuple[int, ...]) -> None:
            name = key.replace(".", "__")
            p = nn.Parameter(torch.empty(*shape))
            nn.init.trunc_normal_(p, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.register_parameter(name, p)
            self._key_to_name[key] = name

        add("token_embeddings.weight", (vocab_size, d_model))
        for i in range(num_layers):
            p = f"layers.{i}."
            add(p + "attn.q_proj.weight", (d_model, d_model))
            add(p + "attn.k_proj.weight", (d_model, d_model))
            add(p + "attn.v_proj.weight", (d_model, d_model))
            add(p + "attn.output_proj.weight", (d_model, d_model))
            add(p + "ln1.weight", (d_model,))
            nn.init.ones_(getattr(self, self._key_to_name[p + "ln1.weight"]))
            add(p + "ln2.weight", (d_model,))
            nn.init.ones_(getattr(self, self._key_to_name[p + "ln2.weight"]))
            add(p + "ffn.w1.weight", (d_ff, d_model))
            add(p + "ffn.w2.weight", (d_model, d_ff))
            if flags.ffn_type == "swiglu":
                add(p + "ffn.w3.weight", (d_ff, d_model))
        add("ln_final.weight", (d_model,))
        nn.init.ones_(getattr(self, self._key_to_name["ln_final.weight"]))
        add("lm_head.weight", (vocab_size, d_model))

    def weight_dict(self) -> dict[str, Tensor]:
        return {k: getattr(self, n) for k, n in self._key_to_name.items()}

    def forward(self, in_indices: Tensor) -> Tensor:
        return transformer_lm_ablation(
            self.vocab_size,
            self.context_length,
            self.d_model,
            self.num_layers,
            self.num_heads,
            self.d_ff,
            self.rope_theta,
            self.flags,
            self.weight_dict(),
            in_indices,
        )
