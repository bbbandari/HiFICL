import enum
from functools import partial
from typing import List, Callable, Dict, Optional, Tuple, Union
import torch
from torch import nn
import re
import torch.utils


class HookType(enum.Enum):
    TEXT_MODEL_LAYER = enum.auto()
    VISION_MODEL_LAYER = enum.auto()


class ShiftStrategy(enum.IntFlag):
    VECTOR_SHIFT = 1
    RECORD_HIDDEN_STATES = 4
    LEARNABLE_SHIFT_SCALE = 8
    MULTI_HEAD = 16
    NLICV_MLP_STATIC_SCALE = 64  # 开启 MLP 模块的静态缩放向量


class BaseHookEncoder(nn.Module):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy(0),
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
    ):
        super().__init__()
        self.attn_strategy = (
            eval(attn_strategy)
            if attn_strategy and eval(attn_strategy)
            else ShiftStrategy(0)
        )
        self.ffn_strategy = (
            eval(ffn_strategy)
            if ffn_strategy and eval(ffn_strategy)
            else ShiftStrategy(0)
        )
        self.lmm = lmm

        if "idefics-9b" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.hidden_size,
                lmm.model.config.num_hidden_layers,
                lmm.model.config.num_attention_heads,
            )
        elif "idefics2-8b" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.text_config.hidden_size,
                lmm.model.config.text_config.num_hidden_layers,
                lmm.model.config.text_config.num_attention_heads,
            )
        elif "llava-interleave" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.text_config.hidden_size,
                lmm.model.config.text_config.num_hidden_layers,
                lmm.model.config.text_config.num_attention_heads,
            )
        else:
            raise ValueError(f"{self.lmm.model_name} is not supported")

        def parse_strategy(prefix, strategy):
            if ShiftStrategy.RECORD_HIDDEN_STATES in getattr(
                self, f"{prefix}_strategy"
            ):
                setattr(
                    self,
                    f"{prefix}_hidden_states",
                    [[] for _ in range(self.lmm_layers)],
                )

            if ShiftStrategy.LEARNABLE_SHIFT_SCALE in strategy and (
                ShiftStrategy.VECTOR_SHIFT not in strategy
            ):
                raise ValueError(
                    "ShiftStrategy.LEARNABLE_SHIFT_SCALE should be used with ShiftStrategy.USE_VECTOR_SHIFT"
                )

        parse_strategy("attn", self.attn_strategy)
        parse_strategy("ffn", self.ffn_strategy)

    def register_hooks(
        self,
        register_fn_name: str,
        targets: List[Union[str, HookType]],
        hooks: Dict[str, Callable],
    ):
        return {
            name: getattr(self.lmm, register_fn_name)(target, hook_fn)
            for target, (name, hook_fn) in zip(targets, hooks.items())
            if hook_fn is not None
        }

    @property
    def decoder_mlp_name(self) -> str:
        if "idefics-9b" in self.lmm.model_name:
            return r"model\.layers\.\d+\.mlp$"
        elif "idefics2-8b" in self.lmm.model_name:
            return r"model\.text_model\.layers\.\d+\.mlp$"
        elif "llava-interleave" in self.lmm.model_name:
            return r"language_model\.model\.layers\.\d+\.mlp$"

    @property
    def decoder_self_attn_name(self) -> str:
        if "idefics-9b" in self.lmm.model_name:
            return r"model\.layers\.\d+\.self_attn$"
        elif "idefics2-8b" in self.lmm.model_name:
            return r"model\.text_model\.layers\.\d+\.self_attn$"
        elif "llava-interleave" in self.lmm.model_name:
            return r"language_model\.model\.layers\.\d+\.self_attn$"

    def register_record_hooks(self, **kwargs):
        # NOTE: record hooks should be registered AFTER all hooks
        def record_hook(m, inputs, outputs, module_name, record_varname, **kwargs):
            layer_idx = int(re.findall(r"\d+", module_name)[0])
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            hidden_states, *_ = outputs
            getattr(self, record_varname)[layer_idx] = hidden_states

        return self.register_hooks(
            "register_forward_hook",
            [
                self.decoder_self_attn_name,
                self.decoder_mlp_name,
            ],
            {
                "attn_record_hook": (
                    partial(record_hook, record_varname="attn_hidden_states")
                    if hasattr(self, "attn_hidden_states")
                    else None
                ),
                "ffn_record_hook": (
                    partial(record_hook, record_varname="ffn_hidden_states")
                    if hasattr(self, "ffn_hidden_states")
                    else None
                ),
            },
        )


class AttnFFNShift(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy(0),
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        shift_scale_init_value=None,
    ):
        """
        Add shift to attention or ffn output. It can also capture hidden states for each layer
        to calculate the layer-wise alignment loss.

        Args:
            lmm: the model to apply shift.
            attn_strategy: the strategy for attention shift.
            ffn_strategy: the strategy for ffn shift.
            shift_scale_init_value: the initial value for the learnable shift scale.
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)

        def parse_strategy(prefix, strategy):
            """
            Create shift modules to ffn output or attention output, based on the strategy.
            """
            if ShiftStrategy.MULTI_HEAD in strategy:
                raise ValueError(
                    f" ShiftStrategy.MULTI_HEAD is not supported, since shift is inserted after {prefix} output"
                )
            if ShiftStrategy.VECTOR_SHIFT in strategy:
                setattr(
                    self,
                    f"{prefix}_shift",
                    torch.nn.Parameter(
                        torch.empty(self.lmm_layers, self.lmm_hidden_dim).normal_(
                            mean=0.0, std=0.01
                        )
                    ),
                )

                if ShiftStrategy.LEARNABLE_SHIFT_SCALE in strategy:
                    setattr(
                        self,
                        f"{prefix}_shift_scale",
                        nn.Parameter(
                            torch.full(
                                [self.lmm_layers],
                                (
                                    shift_scale_init_value
                                    if shift_scale_init_value
                                    else 1.0
                                ),
                            )
                        ),
                    )
                else:
                    self.register_buffer(
                        f"{prefix}_shift_scale", torch.ones(self.lmm_layers)
                    )

        parse_strategy("attn", self.attn_strategy)
        parse_strategy("ffn", self.ffn_strategy)

    def register_shift_hooks(self, **kwargs):
        return self.register_hooks(
            "register_forward_hook",
            [
                self.decoder_self_attn_name,
                self.decoder_mlp_name,
            ],
            {
                "attn_hook": (
                    self._shift_hook("attn") if hasattr(self, "attn_shift") else None
                ),
                "ffn_hook": (
                    self._shift_hook("ffn") if hasattr(self, "ffn_shift") else None
                ),
            },
        )

    def _shift_hook(self, prefix):
        def hook(m, inputs, outputs, module_name, **kwargs):
            layer_idx = int(re.findall(r"\d+", module_name)[0])
            shift = getattr(self, f"{prefix}_shift", None)
            shift_scale = getattr(self, f"{prefix}_shift_scale", None)

            if isinstance(outputs, tuple):
                hidden_states, *rest = outputs
            else:
                hidden_states = outputs

            if shift is not None:
                shift = shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift_scale[layer_idx] * shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

            if isinstance(outputs, tuple):
                return (hidden_states, *rest)
            else:
                return hidden_states

        return hook


# Copied from transformers.models.idefics.modeling_idefics.IdeficsSelfAttention
def idefics_attn_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states=None,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    module_name=None,
    shift_encoder=None,
):
    # if key_value_states are provided this layer is used as a cross-attention layer
    is_cross_attention = self.is_cross_attention or key_value_states is not None
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    if not is_cross_attention:
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
    else:
        _, kv_len, _ = (
            key_value_states.size()
        )  # Note that, in this case, `kv_len` == `kv_seq_len`
        key_states = (
            self.k_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            kv_seq_len += past_key_value[0].shape[-2]
        else:
            kv_seq_len += cache_position[0]

    if not is_cross_attention:
        from transformers.models.idefics.modeling_idefics import apply_rotary_pos_emb

        cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, q_len))
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states) if use_cache else None
        else:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    if self.qk_layer_norms:
        query_states = self.q_layer_norm(query_states)
        key_states = self.k_layer_norm(key_states)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = (
        True if self.is_causal and attention_mask is None and q_len > 1 else False
    )

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # ------------------------- The following part is newly added ---------------------
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )
    # ---------------------------------------------------------------------------------

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    attn_weights = None

    return attn_output, attn_weights, past_key_value


# Copied from transformers.models.mistral.modeling_mistral.MistralSpdaAttention
# The latest version of MistralSpdaAttention is not available in the transformers>=4.46 (not tested)
def idefics2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    module_name=None,
    shift_encoder=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)

    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()

    # ------------------------- The following part is newly added ---------------------
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )
    # ---------------------------------------------------------------------------------

    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention
def llava_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    module_name=None,
    shift_encoder=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings

    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS,
    )

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    # ------------------------- The following part is newly added ---------------------
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )
    # ---------------------------------------------------------------------------------

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


class MultiheadLinear(nn.Module):
    def __init__(self, lmm_num_head, lmm_hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(lmm_num_head, lmm_hidden_dim // lmm_num_head).normal_(0, 0.02)
        )
        self.bias = nn.Parameter(torch.zeros([lmm_num_head]))

    def forward(self, x):
        return torch.einsum("btnd,nd->btn", x, self.weight) + self.bias


class MultiheadProjection(nn.Module):
    def __init__(self, lmm_num_head, lmm_hidden_dim):
        super().__init__()
        head_dim = lmm_hidden_dim // lmm_num_head
        self.weight = nn.Parameter(
            torch.empty(lmm_num_head, head_dim, head_dim).normal_(0, 0.02)
        )
        self.bias = nn.Parameter(torch.zeros([lmm_num_head, head_dim]))

    def forward(self, x):
        return torch.einsum("btnd,ndd->btnd", x, self.weight) + self.bias


class AttnApproxHandle:
    def __init__(self, active=False):
        self.active = active

    def remove(self):
        self.active = False


class AttnApproximator(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
    ):
        """
        The implementation of MimIC attention heads. It train learnable shifts and magnitudes for each layer
        to approximate the in-context demonstrations affected terms (Section 3.2).

        Args:
            lmm: the model to apply shift.
            attn_strategy: the strategy for attention shift.
            ffn_strategy: the strategy for ffn shift.
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]

        if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
            self.log_Z1_lin = nn.ModuleList(
                (
                    MultiheadLinear(self.lmm_num_head, self.lmm_hidden_dim)
                    if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                    else nn.Linear(self.lmm_hidden_dim, 1)
                )
                for _ in range(self.lmm_layers)
            )

        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            # 使用一个ModuleList来为每一层创建一个动态生成shift的投影层
            self.shift_projector = nn.ModuleList(
                (
                    # 如果是多头，使用MultiheadProjection来保持维度一致性
                    MultiheadProjection(self.lmm_num_head, self.lmm_hidden_dim)
                    if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                    # 否则，使用一个简单的线性层
                    else nn.Linear(self.lmm_hidden_dim, self.lmm_hidden_dim)
                )
                for _ in range(self.lmm_layers)
            )

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

    def register_shift_hooks(self, **kwargs):
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                if self.lmm.model_name == "idefics-9b":
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava-interleave" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:

            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])

                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape
        if self.attn_shift_handles[layer_idx].active:
            # [bsz, nh, t, hd] -> [bsz, t, nh, nd]
            query_states_transposed = query_states.transpose(1, 2)

            if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                # [bsz, t, nh, nd] -> [bsz, t, nh * nd]
                query_states_transposed = query_states_transposed.reshape(bsz, t, -1)
                attn_output = attn_output.reshape(bsz, t, -1)

            if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
                # Z1 = \sum{ \exp(x_i X^\top) }
                # calculate Z2 = \sum{ \exp(x_i * \hat{x}^\top) }
                log_Z2 = torch.logsumexp(
                    torch.matmul(query_states, key_states.transpose(-2, -1))
                    / (head_dim**0.5),
                    dim=-1,  # [bsz, nh, t, hd] * [bsz, nh, hd, t] -> [bsz, nh, t, t] -> [bsz, nh, t]
                ).transpose(
                    -2, -1
                )  # [bsz, nh, t] -> [bsz, t, nh]

                if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                    # [bsz, t, nh] -> [bsz, t, 1]
                    log_Z2 = log_Z2.mean(-1, keepdim=True)

                log_Z1 = self.log_Z1_lin[layer_idx](query_states_transposed)

                # shape: [bsz, t, nh] or [bsz, t, 1]
                mu = torch.exp(log_Z1 - torch.logaddexp(log_Z1, log_Z2))
                if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
                    # shape: [bsz, t, nh] -> [bsz, t, nh, 1]
                    mu = mu.unsqueeze(-1)

            if hasattr(self, "shift_projector"): # 检查新的模块是否存在
                # 使用当前层的投影模块，根据query动态生成shift向量
                # 输入 query_states_transposed 的形状是 [bsz, t, nh, nd] 或 [bsz, t, D]
                # 输出 shift 的形状也是 [bsz, t, nh, nd] 或 [bsz, t, D]
                shift = self.shift_projector[layer_idx](query_states_transposed)
                
                if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
                    # shift := SA(q, K_D, V_D) - SA(q, K, V)
                    return (1-mu) * attn_output + mu * shift
            else:
                # never fall in here
                shift = torch.zeros_like(attn_output)

            attn_output = attn_output + shift

        # attn_output: [bsz, t, nh, nd]
        return attn_output


class NLICV_Encoder(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy(0),
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
    ):
        """
        Implementation of the Non-linear In-context Vector (NL-ICV) Encoder.
        This module implements dynamic gating for attention and static affine
        transformation for the MLP module, configured via ShiftStrategy flags.
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]

        # 用于记录协同损失所需的中间输出
        self.mha_outputs_for_syn = []
        self.mlp_outputs_for_syn = []

        # --- 配置 Attention 模块 (动态门控) ---
        # Mimic 的 LEARNABLE_SHIFT_SCALE 标志正好可以用来控制我们的动态门控
        if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
            self.attn_gate_generator = nn.ModuleList(
                (
                    MultiheadLinear(self.lmm_num_head, self.lmm_hidden_dim)
                    if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                    else nn.Linear(self.lmm_hidden_dim, 1)
                )
                for _ in range(self.lmm_layers)
            )

        # 复用 VECTOR_SHIFT 来控制静态上下文向量 v_a
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            self.attn_context_vector = nn.Parameter(
                torch.randn(
                    [self.lmm_layers]
                    + (
                        [self.lmm_num_head, self.lmm_hidden_dim // self.lmm_num_head]
                        if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                        else [self.lmm_hidden_dim]
                    )
                )
                * 0.01
            )

        # --- 配置 FFN/MLP 模块 (静态仿射变换) ---
        # 对应公式: m' = s * m + v_m
        if ShiftStrategy.NLICV_MLP_STATIC_SCALE in self.ffn_strategy:
            # 静态缩放向量 s_m, 初始化为 1
            self.ffn_scale_vector = nn.Parameter(
                torch.ones(self.lmm_layers, self.lmm_hidden_dim)
            )
        
        # 复用 VECTOR_SHIFT 来控制静态偏移向量 v_m
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            # 静态偏移向量 v_m, 初始化为 0
            self.ffn_shift_vector = nn.Parameter(
                torch.zeros(self.lmm_layers, self.lmm_hidden_dim)
            )

    def register_shift_hooks(self, **kwargs):
        """
        注册所有必要的钩子来注入上下文信息。
        """
        # --- 注册 Attention 模块的钩子 (通过替换 forward) ---
        if self.attn_strategy != ShiftStrategy(0):
            if not hasattr(self, "attn_forward_replaced"):
                # (逻辑与 Mimic 相同)
                if "idefics-9b" in self.lmm.model_name:
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava-interleave" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True
        
        registered_hooks = {"attn_hook": self.attn_shift_handles}

        # --- 注册 FFN/MLP 模块的钩子 (实现静态仿射变换) ---
        if self.ffn_strategy != ShiftStrategy(0):
            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])

                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs
                
                # 初始化变换后的状态
                transformed_states = hidden_states

                # 1. 应用静态缩放 (如果开启)
                if ShiftStrategy.NLICV_MLP_STATIC_SCALE in self.ffn_strategy and hasattr(self, 'ffn_scale_vector'):
                    scale = self.ffn_scale_vector[layer_idx][None, None, :]
                    transformed_states = transformed_states * scale
                
                # 2. 应用静态偏移 (如果开启)
                if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy and hasattr(self, 'ffn_shift_vector'):
                    shift = self.ffn_shift_vector[layer_idx][None, None, :]
                    transformed_states = transformed_states + shift

                # 只有在真正进行了变换时才进行范数归一化
                if self.ffn_strategy != ShiftStrategy(0):
                    # (可选但推荐) 保持范数稳定
                    transformed_states = (
                        transformed_states
                        / transformed_states.norm(dim=-1, keepdim=True).clamp(min=1e-6) # 增加 clamp 避免除以0
                        * hidden_states.norm(dim=-1, keepdim=True)
                    )

                # 记录 MLP 输出用于协同损失计算
                if self.training: # 只在训练时记录
                    self.mlp_outputs_for_syn.append(transformed_states)

                if isinstance(outputs, tuple):
                    return (transformed_states, *rest)
                else:
                    return transformed_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        """
        在 Attention 模块内部执行变换的核心函数。
        实现了动态门控融合: a' = g(q) * a + (1-g(q)) * v_a
        """
        if not self.attn_shift_handles[layer_idx].active:
            return attn_output

        # --- 默认行为: 如果没有配置，直接返回原始输出 ---
        final_output = attn_output

        # --- 执行动态门控融合逻辑 ---
        # 必须同时开启动态门控和静态向量
        if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy and ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            head_dim = self.lmm_hidden_dim // self.lmm_num_head
            bsz, nh, t, nd = query_states.shape

            # --- 1. 准备门控 g(q) (逻辑与 Mimic 的 mu 计算类似) ---
            # 形状转换: [bsz, nh, t, hd] -> [bsz, t, nh, nd]
            query_states_transposed = query_states.transpose(1, 2)
            
            # 为了与 Mimic 的多头/单头逻辑兼容
            if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                query_states_transposed = query_states_transposed.reshape(bsz, t, -1)

            # 使用 generator 计算 log_Z1 (对应门控的 logits)
            log_Z1 = self.attn_gate_generator[layer_idx](query_states_transposed)

            # 计算 log_Z2
            log_Z2 = torch.logsumexp(
                torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim**0.5),
                dim=-1,
            ).transpose(-2, -1)
            
            if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                log_Z2 = log_Z2.mean(-1, keepdim=True)

            # 计算门控 gate (mu)
            # 这里使用了 Mimic 的 mu 计算方式，我们直接复用
            gate = torch.exp(log_Z1 - torch.logaddexp(log_Z1, log_Z2))
            if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
                gate = gate.unsqueeze(-1) # -> [bsz, t, nh, 1] for broadcasting

            # --- 2. 获取静态上下文向量 v_a ---
            context_vector = self.attn_context_vector[layer_idx][None, None, :]
            
            # --- 3. 执行融合 ---
            # 形状对齐
            if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                attn_output = attn_output.reshape(bsz, t, -1)
            
            # 这是您修改的核心: return (1-mu) * attn_output + mu * shift
            # 我们这里命名为 gate 和 context_vector
            final_output = (1 - gate) * attn_output + gate * context_vector
            
        # --- (可选) 处理只有静态向量的情况 (类似 LIVE) ---
        elif ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            context_vector = self.attn_context_vector[layer_idx][None, None, :]
            final_output = attn_output + context_vector

        # 记录 MHA 输出用于协同损失计算
        if self.training: # 只在训练时记录
            self.mha_outputs_for_syn.append(final_output)
        
        return final_output


class AttnApproximatorV6(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
    ):
        """
        The implementation of MimIC attention heads with Query-sharing μ (v6).
        This version uses learnable scalar parameters for mu instead of query-dependent computation.
        This is designed to test the importance of query-dependent magnitude.

        Args:
            lmm: the model to apply shift.
            attn_strategy: the strategy for attention shift.
            ffn_strategy: the strategy for ffn shift.
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]

        # --- 修改开始：使用 mu_logits 替代 log_Z1_lin ---
        if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
            # 定义一个可学习的参数 mu，它不依赖于查询 q
            # 我们希望 mu 的值在 (0, 1) 之间，所以用 sigmoid 来约束
            # 为了稳定训练，我们学习 logits，然后通过 sigmoid 得到 mu
            mu_shape = [self.lmm_layers]
            if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
                # 如果是多头，为每一层的每一个头学习一个独立的 mu
                mu_shape.append(self.lmm_num_head)
            else:
                # 否则，为每一层学习一个统一的 mu
                mu_shape.append(1)
            
            # 初始化为0，这样 sigmoid(0) = 0.5，是一个比较中性的起点
            self.mu_logits = nn.Parameter(torch.zeros(mu_shape))
        # --- 修改结束 ---

        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            # 使用一个ModuleList来为每一层创建一个动态生成shift的投影层
            self.shift_projector = nn.ModuleList(
                (
                    # 如果是多头，使用MultiheadProjection来保持维度一致性
                    MultiheadProjection(self.lmm_num_head, self.lmm_hidden_dim)
                    if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                    # 否则，使用一个简单的线性层
                    else nn.Linear(self.lmm_hidden_dim, self.lmm_hidden_dim)
                )
                for _ in range(self.lmm_layers)
            )

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

    def register_shift_hooks(self, **kwargs):
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                if self.lmm.model_name == "idefics-9b":
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava-interleave" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:

            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])

                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape
        if self.attn_shift_handles[layer_idx].active:
            # [bsz, nh, t, hd] -> [bsz, t, nh, nd]
            query_states_transposed = query_states.transpose(1, 2)

            if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                # [bsz, t, nh, nd] -> [bsz, t, nh * nd]
                query_states_transposed = query_states_transposed.reshape(bsz, t, -1)
                attn_output = attn_output.reshape(bsz, t, -1)

            # --- 修改开始：使用 mu_logits 替代复杂的 log_Z1/log_Z2 计算 ---
            if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
                # 从可学习参数中获取 mu_logits
                mu_logits_layer = self.mu_logits[layer_idx] # shape: [nh] or [1]
                # 通过 sigmoid 得到 mu
                mu = torch.sigmoid(mu_logits_layer) # shape: [nh] or [1]
                
                # 为了广播，需要调整 mu 的维度
                if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
                    # 原 shape: [nh]
                    # 目标 shape: [1, 1, nh, 1] 以便和 [bsz, t, nh, nd] 广播
                    mu = mu[None, None, :, None]
                else:
                    # 原 shape: [1]
                    # 目标 shape: [1, 1, 1] 以便和 [bsz, t, hidden_dim] 广播
                    mu = mu[None, None, :]
            # --- 修改结束 ---

            if hasattr(self, "shift_projector"): # 检查新的模块是否存在
                # 使用当前层的投影模块，根据query动态生成shift向量
                # 输入 query_states_transposed 的形状是 [bsz, t, nh, nd] 或 [bsz, t, D]
                # 输出 shift 的形状也是 [bsz, t, nh, nd] 或 [bsz, t, D]
                shift = self.shift_projector[layer_idx](query_states_transposed)
                
                if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
                    # shift := SA(q, K_D, V_D) - SA(q, K, V)
                    return (1-mu) * attn_output + mu * shift
            else:
                # never fall in here
                shift = torch.zeros_like(attn_output)

            attn_output = attn_output + shift

        # attn_output: [bsz, t, nh, nd]
        return attn_output

class AttnApproximatorDirect(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        num_virtual_tokens: int = 16, # 作为超参数，控制虚拟token数量
    ):
        """
        直接近似K_D和V_D的MimIC attention heads实现（v7版本）。
        根据idea4.md的建议，不近似μ和shift，而是直接学习虚拟的K_learn和V_learn来近似演示数据的K_D和V_D。

        Args:
            lmm: 要操作的模型
            attn_strategy: 注意力变换策略
            ffn_strategy: FFN变换策略  
            num_virtual_tokens: 虚拟token数量，默认16
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]
        
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            head_dim = self.lmm_hidden_dim // self.lmm_num_head
            self.num_virtual_tokens = num_virtual_tokens

            # 定义 K_D 的近似 -> self.virtual_keys
            # 维度: [layers, heads, num_v_tokens, head_dim]
            self.virtual_keys = nn.Parameter(
                torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, head_dim) * 0.02
            )
            
            # 定义 V_D 的近似 -> self.virtual_values  
            # 维度: [layers, heads, num_v_tokens, head_dim]
            self.virtual_values = nn.Parameter(
                torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, head_dim) * 0.02
            )

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

    def register_shift_hooks(self, **kwargs):
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                if self.lmm.model_name == "idefics-9b":
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava-interleave" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:

            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])

                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        """
        直接近似K_D和V_D的核心计算逻辑。
        实现公式: Attn_out(q) = (Z2/(Z1+Z2)) * SA(q,K_q,V_q) + (1/(Z1+Z2)) * sum(exp(q·k_i/√d) * v_i)
        
        Args:
            layer_idx: 当前层索引
            query_states: [bsz, nh, t, nd] 查询向量
            key_states: [bsz, nh, t, nd] 查询自身的键向量(K_q)
            attn_output: [bsz, t, nh, nd] 基于查询自身的注意力输出(SA(q,K_q,V_q))
        
        Returns:
            变换后的注意力输出
        """
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape

        if self.attn_shift_handles[layer_idx].active:
            # 1. 计算 Z2 (来自查询自身的注意力分数总和)
            # attn_scores_q: [bsz, nh, t, t]
            attn_scores_q = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim**0.5)
            # log_Z2: [bsz, nh, t]
            log_Z2 = torch.logsumexp(attn_scores_q, dim=-1)

            # 2. 计算 Z1 (来自学习到的虚拟键的分数总和)
            K_learn = self.virtual_keys[layer_idx] # shape: [nh, num_v_tokens, nd]
            # attn_scores_d: [bsz, nh, t, num_v_tokens]
            attn_scores_d = torch.matmul(query_states, K_learn.transpose(-2, -1)) / (head_dim**0.5)
            # log_Z1: [bsz, nh, t]
            log_Z1 = torch.logsumexp(attn_scores_d, dim=-1)

            # 3. 计算总分母 log(Z1 + Z2)
            # log_denominator: [bsz, nh, t]
            log_denominator = torch.logaddexp(log_Z1, log_Z2)

            # 4. 计算第一项: (Z2 / (Z1 + Z2)) * SA(q, K_q, V_q)
            # log_scale_factor_q: [bsz, nh, t]
            log_scale_factor_q = log_Z2 - log_denominator
            scale_factor_q = log_scale_factor_q.exp().transpose(1, 2).unsqueeze(-1) # -> [bsz, t, nh, 1]
            term1 = attn_output * scale_factor_q

            # 5. 计算第二项: (1 / (Z1 + Z2)) * sum(exp(...) * v_i)
            # 使用log-space计算提高数值稳定性
            # log_weights_d: [bsz, nh, t, num_v_tokens]
            log_weights_d = attn_scores_d - log_denominator.unsqueeze(-1)
            weights_d = log_weights_d.exp() # shape: [bsz, nh, t, num_v_tokens]

            V_learn = self.virtual_values[layer_idx] # shape: [nh, num_v_tokens, nd]
            # term2_raw: [bsz, nh, t, nd]
            term2_raw = torch.matmul(weights_d, V_learn)
            term2 = term2_raw.transpose(1, 2) # -> [bsz, t, nh, nd]
            
            # 6. 组合最终结果: Attn_out(q) = Term1 + Term2
            final_output = term1 + term2
            
            return final_output

        # 如果不active，返回原始attn_output
        return attn_output

class AttnApproximatorDirect_v2(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        num_virtual_tokens: int = 16,
    ):
        """
        直接近似K_D和V_D方案的稳定版（依据 idea5 的修改）。
        关键点：取消运行时归一化，使用每层、每头的可学习缩放参数，
        通过 tanh 将缩放因子限制在 (-1, 1) 以提升数值稳定性。
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]
        
        if ShiftStrategy.VECTOR_SHIFT not in self.attn_strategy:
            return

        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        self.num_virtual_tokens = num_virtual_tokens

        # 定义 K_D 和 V_D 的近似参数
        self.virtual_keys = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, head_dim) * 0.02
        )
        self.virtual_values = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, head_dim) * 0.02
        )

        # 根据 idea5：每层、每头使用可学习缩放参数，tanh 限幅，初始化为 0
        self.shift_scaler = nn.Parameter(torch.zeros(self.lmm_layers, self.lmm_num_head))

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

    def register_shift_hooks(self, **kwargs):
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                # (根据你的LMM模型选择正确的 attn_forward 函数)
                # 确保这些函数在当前作用域内是可用的
                if "idefics-9b" in self.lmm.model_name:
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava-interleave" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])

                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape

        if not self.attn_shift_handles[layer_idx].active:
            return attn_output

        # 1. & 2. & 3. 计算 Z1, Z2, 和 log_denominator
        attn_scores_q = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim**0.5)
        log_Z2 = torch.logsumexp(attn_scores_q, dim=-1)
        
        K_learn = self.virtual_keys[layer_idx]
        attn_scores_d = torch.matmul(query_states, K_learn.transpose(-2, -1)) / (head_dim**0.5)
        log_Z1 = torch.logsumexp(attn_scores_d, dim=-1)
        
        log_denominator = torch.logaddexp(log_Z1, log_Z2)
        
        # 4. 计算 term1
        log_scale_factor_q = log_Z2 - log_denominator
        scale_factor_q = log_scale_factor_q.exp().transpose(1, 2).unsqueeze(-1)
        term1 = attn_output * scale_factor_q

        # 5. 计算 term2 (来自学习到的虚拟 V 的贡献)
        log_weights_d = attn_scores_d - log_denominator.unsqueeze(-1)
        weights_d = log_weights_d.exp()
        
        V_learn = self.virtual_values[layer_idx]
        term2_raw = torch.matmul(weights_d, V_learn) # [bsz, nh, t, nd]

        # 根据 idea5：取消归一化，直接用每层、每头的可学习缩放因子进行缩放
        scaler = torch.tanh(self.shift_scaler[layer_idx]).view(1, -1, 1, 1) # [1, nh, 1, 1]
        term2_scaled = term2_raw * scaler

        term2 = term2_scaled.transpose(1, 2) # -> [bsz, t, nh, nd]
        
        # 6. 组合最终结果
        final_output = term1 + term2
        
        return final_output



class AttnApproximatorDirect_final(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        num_virtual_tokens: int = 8,
        lora_rank: int = 4, # K 和 V 使用相同的 rank
    ):
        """
        直接近似方案的性能增强版，同时对 K_learn 和 V_learn 进行低秩分解。
        这不仅通过 V_learn 的分解和初始化保证了训练稳定性，
        还通过 K_learn 的分解引入了正则化，可能提升模型的泛化能力和最终性能。
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]
        
        if ShiftStrategy.VECTOR_SHIFT not in self.attn_strategy:
            return

        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        self.num_virtual_tokens = num_virtual_tokens

        # --- 核心修改 1: 对 K_learn 进行低秩分解 ---
        # K_learn_A: 随机高斯初始化
        self.K_learn_A = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, lora_rank)
        )
        # K_learn_B: 随机高斯初始化
        self.K_learn_B = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, lora_rank, head_dim) * 0.01 # 用较小的标准差
        )
        
        # --- 核心修改 2: 对 V_learn 进行低秩分解 ---
        # V_learn_A: 随机高斯初始化
        self.V_learn_A = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, lora_rank)
        )
        # V_learn_B: 初始化为0，确保训练开始时 shift 为0
        self.V_learn_B = nn.Parameter(
            torch.zeros(self.lmm_layers, self.lmm_num_head, lora_rank, head_dim)
        )
        # --- 修改结束 ---

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

    def register_shift_hooks(self, **kwargs):
        # 这部分代码与你的原始版本完全相同，无需修改
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                # (根据你的LMM模型选择正确的 attn_forward 函数)
                # 确保这些函数在当前作用域内是可用的
                if "idefics-9b" in self.lmm.model_name:
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava-interleave" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])

                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape

        if not self.attn_shift_handles[layer_idx].active:
            return attn_output

        # --- 核心修改: 在运行时从低秩矩阵合成 K_learn 和 V_learn ---
        K_A = self.K_learn_A[layer_idx] # shape: [nh, num_v_tokens, r]
        K_B = self.K_learn_B[layer_idx] # shape: [nh, r, nd]
        K_learn = torch.matmul(K_A, K_B) # shape: [nh, num_v_tokens, nd]
        
        V_A = self.V_learn_A[layer_idx] # shape: [nh, num_v_tokens, r]
        V_B = self.V_learn_B[layer_idx] # shape: [nh, r, nd]
        V_learn = torch.matmul(V_A, V_B) # shape: [nh, num_v_tokens, nd]
        # --- 修改结束 ---

        # 1. & 2. & 3. & 4. (后续的计算逻辑完全不变，因为它们只需要 K_learn 和 V_learn)
        attn_scores_q = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim**0.5)
        log_Z2 = torch.logsumexp(attn_scores_q, dim=-1)
        
        attn_scores_d = torch.matmul(query_states, K_learn.transpose(-2, -1)) / (head_dim**0.5)
        log_Z1 = torch.logsumexp(attn_scores_d, dim=-1)
        
        log_denominator = torch.logaddexp(log_Z1, log_Z2)
        
        log_scale_factor_q = log_Z2 - log_denominator
        scale_factor_q = log_scale_factor_q.exp().transpose(1, 2).unsqueeze(-1)
        term1 = attn_output * scale_factor_q

        # 5. 计算 term2 (shift向量)
        log_weights_d = attn_scores_d - log_denominator.unsqueeze(-1)
        weights_d = log_weights_d.exp()
        
        term2_raw = torch.matmul(weights_d, V_learn)
        
        term2 = term2_raw.transpose(1, 2)
        
        # 6. 组合最终结果
        final_output = term1 + term2
        
        return final_output

class AttnApproximatorDirect_finalv2(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        num_virtual_tokens: int = 8,
        lora_rank: int = 4, # K 和 V 使用相同的 rank
    ):
        """
        直接近似方案的性能增强版，同时对 K_learn 和 V_learn 进行低秩分解。
        这不仅通过 V_learn 的分解和初始化保证了训练稳定性，
        还通过 K_learn 的分解引入了正则化，可能提升模型的泛化能力和最终性能。
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]
        
        if ShiftStrategy.VECTOR_SHIFT not in self.attn_strategy:
            return

        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        self.num_virtual_tokens = num_virtual_tokens

        # --- 核心修改 1: 对 K_learn 进行低秩分解 ---
        # K_learn_A: 随机高斯初始化
        self.K_learn_A = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, lora_rank)
        )
        # K_learn_B: 随机高斯初始化
        self.K_learn_B = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, lora_rank, head_dim) * 0.01 # 用较小的标准差
        )
        
        # --- 核心修改 2: 对 V_learn 进行低秩分解 ---
        # V_learn_A: 随机高斯初始化
        self.V_learn_A = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, lora_rank)
        )
        # V_learn_B: 初始化为0，确保训练开始时 shift 为0
        self.V_learn_B = nn.Parameter(
            torch.zeros(self.lmm_layers, self.lmm_num_head, lora_rank, head_dim)
        )
        # --- 修改结束 ---

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

    def register_shift_hooks(self, **kwargs):
        # 这部分代码与你的原始版本完全相同，无需修改
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                # (根据你的LMM模型选择正确的 attn_forward 函数)
                # 确保这些函数在当前作用域内是可用的
                if "idefics-9b" in self.lmm.model_name:
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava-interleave" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])

                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape

        if not self.attn_shift_handles[layer_idx].active:
            return attn_output

        # --- 核心修改: 在运行时从低秩矩阵合成 K_learn 和 V_learn ---
        K_A = self.K_learn_A[layer_idx] # shape: [nh, num_v_tokens, r]
        K_B = self.K_learn_B[layer_idx] # shape: [nh, r, nd]
        K_learn = torch.matmul(K_A, K_B) # shape: [nh, num_v_tokens, nd]
        
        V_A = self.V_learn_A[layer_idx] # shape: [nh, num_v_tokens, r]
        V_B = self.V_learn_B[layer_idx] # shape: [nh, r, nd]
        V_learn = torch.matmul(V_A, V_B) # shape: [nh, num_v_tokens, nd]
        # --- 修改结束 ---

        # 1. & 2. & 3. & 4. (后续的计算逻辑完全不变，因为它们只需要 K_learn 和 V_learn)
        attn_scores_q = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim**0.5)
        log_Z2 = torch.logsumexp(attn_scores_q, dim=-1)
        
        attn_scores_d = torch.matmul(query_states, K_learn.transpose(-2, -1)) / (head_dim**0.5)
        log_Z1 = torch.logsumexp(attn_scores_d, dim=-1)
        
        log_denominator = torch.logaddexp(log_Z1, log_Z2)
        
        # log_scale_factor_q = log_Z2 - log_denominator
        # scale_factor_q = log_scale_factor_q.exp().transpose(1, 2).unsqueeze(-1)
        # term1 = attn_output * scale_factor_q
        term1 = attn_output

        # 5. 计算 term2 (shift向量)
        log_weights_d = attn_scores_d - log_denominator.unsqueeze(-1)
        weights_d = log_weights_d.exp()
        
        term2_raw = torch.matmul(weights_d, V_learn)
        
        term2 = term2_raw.transpose(1, 2)
        
        # 6. 组合最终结果
        final_output = term1 + term2
        
        return final_output



class AttnApproximatorDirect_finalv3(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        num_virtual_tokens: int = 8,
        lora_rank: int = 4, # 仅 V 使用的 rank
    ):
        """
        直接近似方案的变体，仅对 V_learn 进行低秩分解以保证训练稳定性。
        K_learn 使用完整的矩阵，以探索在无正则化约束下模型的最大表达能力。
        这是一个重要的消融实验设置。
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]
        
        if ShiftStrategy.VECTOR_SHIFT not in self.attn_strategy:
            return

        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        self.num_virtual_tokens = num_virtual_tokens

        # --- 核心修改: K_learn 使用完整的、非分解的矩阵 ---
        # K_learn: 随机高斯初始化，使用较小的标准差以防初始值过大
        self.K_learn = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, head_dim) * 0.01
        )
        # --- 修改结束 ---
        
        # --- V_learn 保持低秩分解不变 ---
        # V_learn_A: 随机高斯初始化
        self.V_learn_A = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, lora_rank)
        )
        # V_learn_B: 初始化为0，确保训练开始时 shift 为0
        self.V_learn_B = nn.Parameter(
            torch.zeros(self.lmm_layers, self.lmm_num_head, lora_rank, head_dim)
        )
        # --- V_learn 部分结束 ---

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

    def register_shift_hooks(self, **kwargs):
        # 这部分代码与你的"final"版本完全相同，因为注册钩子的逻辑没有变
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                # (根据你的LMM模型选择正确的 attn_forward 函数)
                if "idefics-9b" in self.lmm.model_name:
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava" in self.lmm.model_name: # 假设 llava-interleave 也用这个
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            # FFN hook 逻辑保持不变
            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])
                # ... (此处省略与原版相同的FFN hook代码) ...
                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape

        if not self.attn_shift_handles[layer_idx].active:
            return attn_output

        # --- 核心修改: 直接使用完整的 K_learn 矩阵 ---
        K_learn = self.K_learn[layer_idx] # shape: [nh, num_v_tokens, nd]
        # --- 修改结束 ---
        
        # --- V_learn 仍然在运行时从低秩矩阵合成 ---
        V_A = self.V_learn_A[layer_idx] # shape: [nh, num_v_tokens, r]
        V_B = self.V_learn_B[layer_idx] # shape: [nh, r, nd]
        V_learn = torch.matmul(V_A, V_B) # shape: [nh, num_v_tokens, nd]
        # --- V_learn 部分结束 ---

        # 1. & 2. & 3. & 4. (后续的计算逻辑完全不变)
        attn_scores_q = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim**0.5)
        log_Z2 = torch.logsumexp(attn_scores_q, dim=-1)
        
        attn_scores_d = torch.matmul(query_states, K_learn.transpose(-2, -1)) / (head_dim**0.5)
        log_Z1 = torch.logsumexp(attn_scores_d, dim=-1)
        
        log_denominator = torch.logaddexp(log_Z1, log_Z2)
        
        log_scale_factor_q = log_Z2 - log_denominator
        scale_factor_q = log_scale_factor_q.exp().transpose(1, 2).unsqueeze(-1)
        term1 = attn_output * scale_factor_q

        # 5. 计算 term2 (shift向量)
        log_weights_d = attn_scores_d - log_denominator.unsqueeze(-1)
        weights_d = log_weights_d.exp()
        
        term2_raw = torch.matmul(weights_d, V_learn)
        
        term2 = term2_raw.transpose(1, 2)
        
        # 6. 组合最终结果
        final_output = term1 + term2
        
        return final_output



class AttnApproximatorDirect_finalv4(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        num_virtual_tokens: int = 8,
        lora_rank: int = 4, # 仅 K 使用的 rank
    ):
        """
        直接近似方案的变体，仅对 K_learn 进行低秩分解以引入正则化。
        V_learn 使用完整的矩阵，但采用零初始化以保证训练初期的稳定性。
        这个设置用于评估 V_learn 低秩分解对最终性能的影响。
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]
        
        if ShiftStrategy.VECTOR_SHIFT not in self.attn_strategy:
            return

        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        self.num_virtual_tokens = num_virtual_tokens

        # --- K_learn 保持低秩分解不变 ---
        # K_learn_A: 随机高斯初始化
        self.K_learn_A = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, lora_rank)
        )
        # K_learn_B: 随机高斯初始化
        self.K_learn_B = nn.Parameter(
            torch.randn(self.lmm_layers, self.lmm_num_head, lora_rank, head_dim) * 0.01 # 用较小的标准差
        )
        # --- K_learn 部分结束 ---
        
        # --- 核心修改: V_learn 使用完整的、非分解的矩阵 ---
        # V_learn: 初始化为零矩阵，以确保训练开始时 shift 为0，保证稳定性。
        self.V_learn = nn.Parameter(
            torch.zeros(self.lmm_layers, self.lmm_num_head, self.num_virtual_tokens, head_dim)
        )
        # --- 修改结束 ---

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

    def register_shift_hooks(self, **kwargs):
        # 这部分代码与你的"final"版本完全相同
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                # (根据你的LMM模型选择正确的 attn_forward 函数)
                if "idefics-9b" in self.lmm.model_name:
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            # FFN hook 逻辑保持不变
            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])
                # ... (此处省略与原版相同的FFN hook代码) ...
                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape

        if not self.attn_shift_handles[layer_idx].active:
            return attn_output

        # --- K_learn 仍然在运行时从低秩矩阵合成 ---
        K_A = self.K_learn_A[layer_idx] # shape: [nh, num_v_tokens, r]
        K_B = self.K_learn_B[layer_idx] # shape: [nh, r, nd]
        K_learn = torch.matmul(K_A, K_B) # shape: [nh, num_v_tokens, nd]
        # --- K_learn 部分结束 ---
        
        # --- 核心修改: 直接使用完整的 V_learn 矩阵 ---
        V_learn = self.V_learn[layer_idx] # shape: [nh, num_v_tokens, nd]
        # --- 修改结束 ---

        # 1. & 2. & 3. & 4. (后续的计算逻辑完全不变)
        attn_scores_q = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim**0.5)
        log_Z2 = torch.logsumexp(attn_scores_q, dim=-1)
        
        attn_scores_d = torch.matmul(query_states, K_learn.transpose(-2, -1)) / (head_dim**0.5)
        log_Z1 = torch.logsumexp(attn_scores_d, dim=-1)
        
        log_denominator = torch.logaddexp(log_Z1, log_Z2)
        
        log_scale_factor_q = log_Z2 - log_denominator
        scale_factor_q = log_scale_factor_q.exp().transpose(1, 2).unsqueeze(-1)
        term1 = attn_output * scale_factor_q

        # 5. 计算 term2 (shift向量)
        log_weights_d = attn_scores_d - log_denominator.unsqueeze(-1)
        weights_d = log_weights_d.exp()
        
        term2_raw = torch.matmul(weights_d, V_learn)
        
        term2 = term2_raw.transpose(1, 2)
        
        # 6. 组合最终结果
        final_output = term1 + term2
        
        return final_output