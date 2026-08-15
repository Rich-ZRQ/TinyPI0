"""Attention-mask construction used by pi0."""

import torch
from torch import Tensor


def make_att_2d_masks(
    pad_masks: Tensor,
    att_masks: Tensor,
) -> Tensor:
    """Build block-causal attention masks.

    Args:
        pad_masks:
            Boolean tensor [B, N]. True means the token is valid.

        att_masks:
            Boolean tensor [B, N]. True means this token starts a
            new causal block. False means it shares the previous
            token's block.

    Returns:
        Boolean tensor [B, N, N].

        Dimension 1 is the query-token dimension.
        Dimension 2 is the key-token dimension.
    """

    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must have shape [B, N], got {pad_masks.shape}")

    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must have shape [B, N], got {att_masks.shape}")

    if pad_masks.shape != att_masks.shape:
        raise ValueError(
            f"pad_masks and att_masks must have the same shape, got {pad_masks.shape} and {att_masks.shape}"
        )

    if pad_masks.dtype != torch.bool:
        raise TypeError(f"pad_masks must be bool, got {pad_masks.dtype}")

    if att_masks.dtype != torch.bool:
        raise TypeError(f"att_masks must be bool, got {att_masks.dtype}")

    if pad_masks.device != att_masks.device:
        raise ValueError("pad_masks and att_masks must be on the same device")

    block_ids = torch.cumsum(
        att_masks.to(torch.int64),
        dim=1,
    )

    key_block_ids = block_ids[:, None, :]  # [B, 1, N]  把块号摆成一行,代表所有 key 的块号。
    query_block_ids = block_ids[:, :, None]  # [B, N, 1]  把块号摆成一列,代表所有 query 的块号。

    causal_mask = (
        key_block_ids <= query_block_ids
    )  # 广播成 [B, N, N]   query_block_ids >= key_block_ids, 表示query在块i只能看到块i及之前的块，不能看到之后的
    """
    key 块号 →
              1    2    3    4
  query  1 [ ✓    ✗    ✗    ✗ ]   query在块1,只能看块≤1
  块号   2 [ ✓    ✓    ✗    ✗ ]   query在块2,能看块1、2
   ↓     3 [ ✓    ✓    ✓    ✗ ]
         4 [ ✓    ✓    ✓    ✓ ]   query在块4,能看全部
    """

    key_is_valid = pad_masks[:, None, :]  # [B, 1, N]
    query_is_valid = pad_masks[:, :, None]  # [B, N, 1]
    valid_mask = key_is_valid & query_is_valid  # 广播成 [B, N, N]   query和key都必须是有效的token，才能互相看到

    return causal_mask & valid_mask  # 最终实现块结构上允许(块因果/块内双向) 且 两端都是真实 token(非 padding)
