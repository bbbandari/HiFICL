# ==================== 新增: 基于 Qwen2Attention 实现的 llava_qwen_attn_forward ====================
# ==================== 真正的最终解决方案 ====================
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

    # `attn_output` is a multi-head tensor, e.g., [bsz, nh, t, nd]
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
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


# ===========================MHA==================================
def qwen2_attn_forward(
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

    from transformers.models.qwen2.modeling_qwen2 import (
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