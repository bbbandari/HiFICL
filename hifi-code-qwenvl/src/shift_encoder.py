import enum
from functools import partial
from typing import List, Callable, Dict, Optional, Tuple, Union
try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
import torch
from torch import nn
import re
import torch.utils

# 导入 transformers 相关类型

from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import FlashAttentionKwargs


from dataclasses import dataclass

# 导入类型提示所需的类
from transformers.cache_utils import Cache
from transformers.utils import is_flash_attn_2_available


def find_module_prefix(model, target_module_name: str) -> str:
    """
    Searches through the model's named modules to find the full path prefix
    for a target module.
    
    Args:
        model: The model to search within (e.g., self.lmm).
        target_module_name: A unique part of the target module's name, 
                            e.g., "layers.0.self_attn".

    Returns:
        The full prefix needed to match the module, e.g., "base_model.model.language_model.model."
    """
    candidates = []
    for name, module in model.named_modules():
        if name.endswith(target_module_name):
            # We found our target module. The prefix is the full name minus the target part.
            prefix = name[: -len(target_module_name)]
            candidates.append((name, prefix))
    
    if not candidates:
        return None
    
    # Prefer language model modules over vision model modules
    for name, prefix in candidates:
        if 'language_model' in name and 'vision_tower' not in name:
            return prefix
    
    # If no language model found, return the first match
    name, prefix = candidates[0]
    return prefix


def get_base_language_model(model):
    """
    Recursively unwraps a model to find the base language model.
    Handles PEFT wrapping and other common model compositions.
    
    Args:
        model: The model to unwrap (could be wrapped by PEFT, LLaVA, etc.)
        
    Returns:
        The base language model object
        
    Raises:
        AttributeError: If no base language model can be found
    """
    # Case 1: Direct language_model attribute (e.g., LLaVA without PEFT)
    if hasattr(model, "language_model"):
        return model.language_model
    
    # Case 2: PEFT wrapping - check base_model
    if hasattr(model, "base_model"):
        return get_base_language_model(model.base_model)
    
    # Case 3: Common wrapping pattern - check model attribute
    if hasattr(model, "model"):
        return get_base_language_model(model.model)
    
    # Case 4: The model itself might be the language model
    # Check for typical language model attributes
    if hasattr(model, 'layers') and hasattr(model, 'embed_tokens'):
        return model
    
    # Case 5: Check for config attribute (might be a config object)
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        # This might be a config object, not the actual model
        # We need to go back up the chain
        pass
    
    raise AttributeError(f"Could not find the base language model after unwrapping. "
                        f"Model type: {type(model)}, "
                        f"Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")


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
        # Handle both string and ShiftStrategy object inputs
        if isinstance(attn_strategy, str):
            self.attn_strategy = eval(attn_strategy)
        else:
            self.attn_strategy = (
                attn_strategy
                if attn_strategy
                else ShiftStrategy(0)
            )
            
        if isinstance(ffn_strategy, str):
            self.ffn_strategy = eval(ffn_strategy)
        else:
            self.ffn_strategy = (
                ffn_strategy
                if ffn_strategy
                else ShiftStrategy(0)
            )
        self.lmm = lmm

        # --- NEW ROBUST INITIALIZATION ---
        try:
            # First, get the core language model, regardless of wrapping
            # We start from lmm.model as lmm is likely a higher-level wrapper
            language_model = get_base_language_model(self.lmm.model)
            
            # Now, get the config from the unwrapped language model
            # For most models (Qwen, Llama, etc.), the config is directly accessible
            if hasattr(language_model, 'config'):
                config = language_model.config
            else:
                # Fallback: try to get config from the original model structure
                # This handles cases where the config is stored differently
                if "idefics2-8b" in self.lmm.model_name or "llava-v1.6" in self.lmm.model_name:
                    # Idefics2 and LLaVA store config in text_config
                    config = self.lmm.model.config.text_config
                else:
                    raise AttributeError("Unwrapped language model does not have a 'config' attribute.")

            # Extract the parameters using consistent naming
            # Most modern models use these standard names
            self.lmm_hidden_dim = config.hidden_size
            self.lmm_layers = config.num_hidden_layers
            self.lmm_num_head = config.num_attention_heads

        except AttributeError as e:
            print(f"Error during model config initialization: {e}")
            print("Falling back to legacy initialization method...")
            
            # Fallback to old method for debugging and compatibility
            if "idefics-9b" in self.lmm.model_name:
                config = lmm.model.config
                self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                    config.hidden_size, config.num_hidden_layers, config.num_attention_heads
                )
            elif "idefics2-8b" in self.lmm.model_name:
                config = lmm.model.config.text_config
                self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                    config.hidden_size, config.num_hidden_layers, config.num_attention_heads
                )
            elif "llava-v1.6" in self.lmm.model_name:
                config = lmm.model.config.text_config
                self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                    config.hidden_size, config.num_hidden_layers, config.num_attention_heads
                )
            else:
                raise ValueError(f"Failed to automatically determine config for {self.lmm.model_name}. "
                               f"This suggests the model structure is different than expected. "
                               f"Error: {e}")
        # --- END OF NEW INITIALIZATION ---

        # --- DYNAMIC PATH GENERATION ---
        # 为了健壮性，我们尝试几种可能的锚点
        possible_attn_anchors = [
            "language_model.layers.0.self_attn",  # LLaVA models (most common)
            "base_model.model.layers.0.self_attn",
            "language_model.model.layers.0.self_attn",
            "text_model.layers.0.self_attn",
            "base_model.language_model.layers.0.self_attn",
            "base_model.model.language_model.layers.0.self_attn",
            "base_model.model.model.language_model.layers.0.self_attn",  # Qwen3-VL with PEFT
            "layers.0.self_attn",
        ]
        possible_mlp_anchors = [
            "language_model.layers.0.mlp",  # LLaVA models (most common)
            "base_model.model.layers.0.mlp",
            "language_model.model.layers.0.mlp",
            "text_model.layers.0.mlp",
            "base_model.language_model.layers.0.mlp",
            "base_model.model.language_model.layers.0.mlp",
            "base_model.model.model.language_model.layers.0.mlp",  # Qwen3-VL with PEFT
            "layers.0.mlp",
        ]

        # For Qwen3-VL models, use a special handling approach
        if "Qwen3-VL" in self.lmm.model_name:
            # For Qwen3-VL models, handle both PEFT and non-PEFT cases
            if hasattr(self.lmm.model, 'base_model') and hasattr(self.lmm.model.base_model, 'model'):
                # PEFT wrapped case: base_model.model.model.language_model.layers.X.self_attn
                self._attn_prefix = "base_model.model.model.language_model."
                self._mlp_prefix = "base_model.model.model.language_model."
            elif hasattr(self.lmm.model, 'language_model'):
                # Direct case: model.language_model.layers.X.self_attn
                self._attn_prefix = "language_model."
                self._mlp_prefix = "language_model."
            else:
                # Fallback to search
                search_model = self.lmm.model
                self._attn_prefix = None
                for anchor in possible_attn_anchors:
                    prefix = find_module_prefix(search_model, anchor)
                    if prefix is not None:
                        self._attn_prefix = prefix
                        break
                self._mlp_prefix = None
                for anchor in possible_mlp_anchors:
                    prefix = find_module_prefix(search_model, anchor)
                    if prefix is not None:
                        self._mlp_prefix = prefix
                        break
        # For LLaVA models, use a special handling approach
        elif hasattr(self.lmm.model, 'language_model'):
            # For LLaVA models, we know the structure is language_model.layers.X.self_attn
            # So we can directly set the prefix without searching
            self._attn_prefix = "language_model."
            self._mlp_prefix = "language_model."
        else:
            # For other models, use the original search approach
            search_model = self.lmm.model
            
            self._attn_prefix = None
            for anchor in possible_attn_anchors:
                prefix = find_module_prefix(search_model, anchor)
                if prefix is not None:
                    self._attn_prefix = prefix
                    break

            self._mlp_prefix = None
            for anchor in possible_mlp_anchors:
                prefix = find_module_prefix(search_model, anchor)
                if prefix is not None:
                    self._mlp_prefix = prefix
                    break

        if self._attn_prefix is None or self._mlp_prefix is None:
            raise ValueError(f"Failed to find module prefixes for {self.lmm.model_name}. Check the paths.")

        # --- END OF DYNAMIC PATH GENERATION ---

        def parse_strategy(prefix, strategy):
            if ShiftStrategy.RECORD_HIDDEN_STATES in strategy:
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
        """
        为 LLaVA 类模型提供一个稳健的、与包装无关的正则表达式。
        """
        # 通过模型名称来判断是否为 LLaVA 类模型，这是一种简单有效的方法
        model_name = getattr(self.lmm, "model_name", "").lower()
        is_llava_like = "llava" in model_name or "interleave" in model_name or "qwen3-vl" in model_name

        if is_llava_like:
            # 直接使用最稳健的正则表达式，它能处理任何前缀
            return r"^(?!.*vision_tower).*language_model\.layers\.\d+\.mlp$"
        else:
            # 对于其他模型，保留原始的基于前缀的逻辑
            prefix_regex = self._mlp_prefix.replace('.', r'\.')
            return f"^(?!.*vision_tower).*{prefix_regex}layers\\.\\d+\\.mlp$"

    @property
    def decoder_self_attn_name(self) -> str:
        """
        为 LLaVA 类模型提供一个稳健的、与包装无关的正则表达式。
        """
        model_name = getattr(self.lmm, "model_name", "").lower()
        is_llava_like = "llava" in model_name or "interleave" in model_name or "qwen3-vl" in model_name

        if is_llava_like:
            # 直接使用最稳健的正则表达式，它能处理任何前缀
            return r"^(?!.*vision_tower).*language_model\.layers\.\d+\.self_attn$"
        else:
            # 对于其他模型，保留原始的基于前缀的逻辑
            prefix_regex = self._attn_prefix.replace('.', r'\.')
            return f"^(?!.*vision_tower).*{prefix_regex}layers\\.\\d+\\.self_attn$"

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


def mistral_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    module_name=None,
    shift_encoder=None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

    input_shape = hidden_states.shape[:-1]
    bsz, q_len, _ = hidden_states.size()
    
    # --- 准备 Q, K, V (无变化) ---
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings

    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS,
        repeat_kv,
    )

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
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

    return attn_output, None

 # ==================== 新增: 基于 Qwen2Attention 实现的 llava_qwen_attn_forward ====================
# Based on transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward
# and adapted to be a replaceable forward function like idefics2_attn_forward.
def qwen2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    # 1. 使用正确的接口规范
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    module_name=None,
    shift_encoder=None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

    # --- Part 1: Projections, RoPE, KV Cache (100% from source) ---
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 2. 使用正确的依赖
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, repeat_kv
    )

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # --- Part 2: Attention Calculation (100% from source) ---
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # Prepare kwargs for attention interface, handle sliding_window if available
    attn_kwargs = {
        "dropout": 0.0 if not self.training else self.attention_dropout,
        "scaling": self.scaling,
        **kwargs,
    }

    # Only add sliding_window if the attention layer has it (Qwen2 has it, Qwen3-VL doesn't)
    if hasattr(self, 'sliding_window'):
        attn_kwargs["sliding_window"] = self.sliding_window

    # `attn_output` is a multi-head tensor, e.g., [bsz, nh, t, nd]
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        **attn_kwargs,
    )

    # ========================== INJECTION BLOCK ==========================
    # 3. 正确的数据流：直接在多头输出上操作
    layer_idx = int(re.findall(r"\d+", module_name)[0]) # Use the robust way to get layer_idx
    repeated_key_states = repeat_kv(key_states, self.num_key_value_groups)

    # `do_shift` directly modifies the multi-head `attn_output`
    shifted_attn_output = shift_encoder.do_shift(
        layer_idx,
        query_states,
        repeated_key_states,
        attn_output # Pass the multi-head tensor directly
    )

    attn_output = shifted_attn_output
    # =====================================================================

    # --- Part 3: Final Projection and Return (100% from source) ---
    # The original reshape now correctly packs our modified multi-head tensor
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

def qwen3_vl_text_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    # --- 自定义参数 ---
    module_name: str | None = None,
    shift_encoder: any = None,
    # ------------------
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    
    # 1. 导入 Qwen3-VL 特有的 RoPE 计算函数
    # 注意：这个路径取决于你 transformers 的安装位置，通常是 transformers.models.qwen3_vl.modeling_qwen3_vl
    # 如果 transformers 库里还没发布 qwen3_vl，这里应该是你本地文件的路径
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb,  # Qwen3VL 有自己的 RoPE 实现，必须从这里导
    )

    # 2. 导入通用的 Attention 工具函数
    # Qwen 系列通常复用 Qwen2 的逻辑，或者在 Qwen3VL 文件中定义了别名
    # 我们先尝试从 Qwen3VL 导，如果报错（比如没导出），就从 Qwen2 导
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            eager_attention_forward,
            ALL_ATTENTION_FUNCTIONS,
        )
    except ImportError:
        # 如果 Qwen3VL 文件里没有显式导出这些，通常意味着它直接用了 Qwen2 的
        from transformers.models.qwen2.modeling_qwen2 import (
            eager_attention_forward,
            ALL_ATTENTION_FUNCTIONS,
        )

    # 3. 导入 repeat_kv (用于处理 GQA/MQA)
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
    except ImportError:
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    # ========================== 原生逻辑开始 ==========================
    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # 投影和 Norm (不需要改，self 里都有)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 应用 RoPE (使用导入的 apply_rotary_pos_emb)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # KV Cache 更新
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 选择 Attention 实现方式 (SDPA / Eager / FlashAttn)
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        # 使用导入的 ALL_ATTENTION_FUNCTIONS
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # 计算 Attention
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    # ========================== [注入点] ==========================
    if shift_encoder is not None and module_name is not None:
        # 解析层号
        layer_idx = int(re.findall(r"\d+", module_name)[0])
        
        # 处理 Group Query Attention (GQA) 的维度对齐
        # shift_encoder 通常期望 key/value 的头数和 query 一样多
        if self.num_key_value_groups > 1:
            key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        else:
            key_states_expanded = key_states

        # 执行 Shift 操作
        # 输入: [bs, num_heads, seq_len, head_dim]
        shifted_attn_output = shift_encoder.do_shift(
            layer_idx,
            query_states,
            key_states_expanded,
            attn_output
        )
        attn_output = shifted_attn_output
    # =============================================================

    # 输出投影
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
                elif "llava-v1.6" in self.lmm.model_name:
                    new_attn_foward = mistral_attn_forward
                elif "llava-interleave" in self.lmm.model_name or "interleave" in self.lmm.model_name:
                    new_attn_foward = qwen2_attn_forward
                elif "Qwen3-VL" in self.lmm.model_name:
                    new_attn_foward = qwen3_vl_text_attn_forward
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
                elif "llava-v1.6" in self.lmm.model_name:
                    new_attn_foward = mistral_attn_forward
                elif "llava-interleave" in self.lmm.model_name or "interleave" in self.lmm.model_name:
                    new_attn_foward = qwen2_attn_forward
                elif "Qwen3-VL" in self.lmm.model_name:
                    new_attn_foward = qwen3_vl_text_attn_forward
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