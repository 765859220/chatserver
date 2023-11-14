import copy
from itertools import chain
from typing import Callable, Optional, Union

import torch
from torch import nn

from .check_utils import check_types


def get_device(obj: Union[torch.Tensor, nn.Module]) -> torch.device:
    """get device of torch obj, including tensor and module"""
    check_types(obj, (torch.Tensor, nn.Module))
    if hasattr(obj, "device"):
        return obj.device
    try:
        return next(chain(obj.parameters(), obj.buffers())).device
    except StopIteration:
        return torch.device("cpu")


def _reduce(
    x: torch.Tensor,
    op: Callable,
    excluded_axis: Optional[int] = None,
    keepdim=False,
) -> torch.Tensor:
    """reduce the input tensor with given op, excluding the axis(if given)"""
    if excluded_axis is None:
        return op(x)
    dims = [i for i in range(x.ndim) if i != excluded_axis]
    ret = x
    for dim in dims:
        ret = op(ret, dim=dim, keepdim=True)[0]
    if not keepdim:
        ret = torch.reshape(ret, (ret.shape[excluded_axis],))
    return ret


def reduce_max(
    x: torch.Tensor, excluded_axis: Optional[int] = None, keepdim=False
) -> torch.Tensor:
    return _reduce(x, torch.max, excluded_axis, keepdim)


def reduce_min(
    x: torch.Tensor, excluded_axis: Optional[int] = None, keepdim=False
) -> torch.Tensor:
    return _reduce(x, torch.min, excluded_axis, keepdim)


def fuse_conv_bn_weights(
    conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False
):
    """fuse conv weight"""
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(shape)
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return conv_w, conv_b


def fuse_linear_bn_weights(
    linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b
):
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)

    fused_w = linear_w * bn_scale.unsqueeze(-1)
    fused_b = (linear_b - bn_rm) * bn_scale + bn_b

    return fused_w, fused_b


def fuse_conv_bn(
    conv: nn.Conv2d, bn: nn.BatchNorm2d, transpose: bool = False
) -> nn.Conv2d:
    """fuse conv and bn, return a new conv module"""
    fused_conv = copy.deepcopy(conv)

    weight, bias = fuse_conv_bn_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        transpose,
    )

    fused_conv.weight = nn.Parameter(weight.data)
    fused_conv.bias = nn.Parameter(bias.data)
    return fused_conv


def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """fuse linear and bn, return a new linear module"""
    fused_linear = copy.deepcopy(linear)

    weight, bias = fuse_linear_bn_weights(
        fused_linear.weight,
        fused_linear.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )
    fused_linear.weight = nn.Parameter(weight)
    fused_linear.bias = nn.Parameter(bias)

    return fused_linear
