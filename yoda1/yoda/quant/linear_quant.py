from typing import Tuple

import torch


@torch.no_grad()
def get_scale_zp_for_sym_quant(
    x_max: torch.Tensor, max_bound: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """get scale and zp for symmetric quantization

    float | quantized
    x_max -> max_bound

    x_float = x_q * s
    => s = x_max / max_bound, zp = 0
    """
    s = x_max / max_bound
    zp = torch.zeros_like(s)
    return s, zp


@torch.no_grad()
def get_scale_zp_for_asym_quant(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    min_bound: int,
    max_bound: int,
    zero_in_the_range: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """get scale and zp for asymmetric quantization

    float |  quantized
    x_min -> min_bound
    x_max -> max_bound

    x_float = (x_q - zp) * s

    => s = (x_max - x_min) / (max_bound - min_bound)
    """
    if zero_in_the_range:
        x_min = torch.where(x_min < 0, x_min, torch.zeros_like(x_min))
        x_max = torch.where(x_max > 0, x_max, torch.zeros_like(x_max))
    s = (x_max - x_min) / (max_bound - min_bound)
    zp = min_bound - x_min / s
    # find the nearest zp
    zp = torch.round(zp)
    return s, zp


def ste(x: torch.Tensor) -> torch.Tensor:
    """ste"""
    y = x.detach()
    return torch.round(y) + x - y


def _fake_linear_quant(
    x: torch.Tensor,
    s: torch.Tensor,
    zp: torch.Tensor,
    min_bound: int,
    max_bound: int,
) -> torch.Tensor:
    """fake linear quant

    x_q = round(x / scale) + zp
    x_out = (x_q - zp) * scale
    """
    xscaled = x / s + zp
    xq = ste(xscaled).clamp_(min_bound, max_bound)
    xout = (xq - zp) * s
    return xout


def fake_linear_quant_per_tensor(
    x: torch.Tensor,
    s: torch.Tensor,
    zp: torch.Tensor,
    min_bound: int,
    max_bound: int,
) -> torch.Tensor:
    """fake linear quant per tensor"""
    return _fake_linear_quant(x, s, zp.float(), min_bound, max_bound)


def fake_linear_quant_per_axis(
    x: torch.Tensor,
    s: torch.Tensor,
    zp: torch.Tensor,
    axis: int,
    min_bound: int,
    max_bound: int,
) -> torch.Tensor:
    """fake linear quant per axis"""
    # broadcast s/zp
    shape = [1 for _ in x.shape]
    shape[axis] = s.numel()
    s = torch.broadcast_to(s, shape)
    zp = torch.broadcast_to(zp, shape)
    return _fake_linear_quant(x, s, zp.float(), min_bound, max_bound)
