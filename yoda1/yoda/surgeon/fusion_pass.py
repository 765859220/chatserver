from typing import Callable, Tuple, Type

import torch
from torch import nn

from yoda.utils import torch_utils

ModuleType = Type[nn.Module]


def _fuse(
    model: nn.Module,
    type1: Tuple[ModuleType],
    type2: Tuple[ModuleType],
    fuse_helper: Callable,
) -> nn.Module:
    last_conv = None
    last_conv_name = None
    for name, child in model.named_children():
        if isinstance(child, type1):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = fuse_helper(last_conv, child)
            model._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            model._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, type2):
            last_conv = child
            last_conv_name = name
        else:
            _fuse(child, type1, type2, fuse_helper)
    return model


def fuse_conv_bn_pass(model: nn.Module) -> nn.Module:
    """fuse conv bn in the module"""
    return _fuse(
        model,
        (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm),
        (nn.Conv2d,),
        torch_utils.fuse_conv_bn,
    )


def fuse_linear_bn_pass(model: nn.Module) -> nn.Module:
    """fuse linear bn in the module"""
    return _fuse(
        model,
        (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm),
        (nn.Linear,),
        torch_utils.fuse_linear_bn,
    )
