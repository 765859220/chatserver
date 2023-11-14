from typing import Optional

import torch
import triton
from triton import language as tl
from triton.language.math import llrint

from .utils import autotune


@triton.jit
def _fake_linear_quant_per_tensor_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    min_bound,
    max_bound,
    n_element,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_element
    x = tl.load(x_ptr + offset, mask)
    s = tl.load(scale_ptr, mask=None)
    zp = tl.load(zp_ptr, mask=None)

    x_scaled = x / s + zp
    x_quant = llrint(x_scaled)
    x_quant = tl.where(x_quant > min_bound, x_quant, min_bound)
    x_quant = tl.where(x_quant < max_bound, x_quant, max_bound)

    out = (x_quant - zp) * s
    tl.store(out_ptr + offset, out, mask)


@triton.jit
def _fake_linear_quant_per_axis_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    min_bound,
    max_bound,
    C,
    H,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid = tl.program_id(0)
    cidx = pid % C

    offset = tl.arange(0, BLOCK_SIZE_H)
    x = tl.load(x_ptr + pid * H + offset, mask=offset < H)
    s = tl.load(scale_ptr + cidx, mask=None)
    zp = tl.load(zp_ptr + cidx, mask=None)

    x_scaled = x / s + zp
    x_quant = llrint(x_scaled)
    x_quant = tl.where(x_quant > min_bound, x_quant, min_bound)
    x_quant = tl.where(x_quant < max_bound, x_quant, max_bound)

    out = (x_quant - zp) * s
    tl.store(out_ptr + pid * H + offset, out, mask=offset < H)


def fake_linear_quant(
    x: torch.Tensor,
    s: torch.Tensor,
    zp: torch.Tensor,
    min_bound: int,
    max_bound: int,
    axis: Optional[int] = None,
) -> torch.Tensor:
    """fake linear quant per tensor"""
    assert x.is_cuda and s.is_cuda and zp.is_cuda, "input not in cuda"
    assert s.numel() == zp.numel()
    out = torch.empty_like(x)
    if s.numel() == 1:
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        _fake_linear_quant_per_tensor_kernel[grid](
            x,
            s,
            zp,
            out,
            min_bound,
            max_bound,
            x.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        assert axis is not None
        H = x.stride(axis)
        C = x.shape[axis]
        BLOCK_SIZE = triton.next_power_of_2(H)
        _fake_linear_quant_per_axis_kernel[(x.numel() // H,)](
            x, s, zp, out, min_bound, max_bound, C, H, BLOCK_SIZE
        )
    return out
