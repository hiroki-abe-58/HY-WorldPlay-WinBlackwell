# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from einops import rearrange
import torch
import torch.nn.functional as F

# Windows/Blackwell compatibility: flash_attn may not be available
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_varlen_qkvpacked_func = None
    pad_input = None
    unpad_input = None


def _torch_attention_fallback(q, k, v, attn_mask=None, dropout_p=0.0, causal=False):
    """
    PyTorch native scaled_dot_product_attention fallback for Windows.
    """
    # q, k, v: [B, S, H, D] -> [B, H, S, D] for SDPA
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    
    out = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
    )
    
    # [B, H, S, D] -> [B, S, H, D]
    out = out.transpose(1, 2)
    return out


def flash_attn_no_pad(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    """
    Compute attention with flash_attn or fallback to PyTorch SDPA.
    
    Windows/Blackwell compatibility: If flash_attn is not available,
    uses PyTorch's native scaled_dot_product_attention.
    """
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    head_dim = qkv.shape[-1]
    
    # Fallback to PyTorch native attention if flash_attn is not available
    if not FLASH_ATTN_AVAILABLE:
        # qkv: [B, S, 3, H, D] -> split into q, k, v
        q, k, v = qkv.unbind(dim=2)  # Each: [B, S, H, D]
        
        # Handle attention mask for SDPA
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] with True for valid positions
            # SDPA expects: [B, 1, 1, S] or [B, 1, S, S]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            # Convert to additive mask: 0 for valid, -inf for invalid
            attn_mask = torch.where(mask, 0.0, float('-inf')).to(q.dtype)
        
        output = _torch_attention_fallback(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, causal=causal)
        return output
    
    # Original flash_attn implementation
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad_v3(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    """
    FlashAttention V3 variant with fallback to PyTorch SDPA.
    
    Windows/Blackwell compatibility: If flash_attn is not available,
    uses PyTorch's native scaled_dot_product_attention.
    """
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    
    # Fallback to PyTorch native attention if flash_attn is not available
    if not FLASH_ATTN_AVAILABLE:
        q, k, v = qkv.unbind(dim=2)  # Each: [B, S, H, D]
        
        # Handle attention mask for SDPA
        attn_mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(mask, 0.0, float('-inf')).to(q.dtype)
        
        output = _torch_attention_fallback(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, causal=causal)
        return output
    
    # Try flash_attn V3
    try:
        from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
    except ImportError:
        # Fallback to PyTorch if V3 not available
        q, k, v = qkv.unbind(dim=2)
        attn_mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(mask, 0.0, float('-inf')).to(q.dtype)
        output = _torch_attention_fallback(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, causal=causal)
        return output

    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
    )

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output
