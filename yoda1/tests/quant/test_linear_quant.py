import pytest
import torch

from yoda.quant.linear_quant import (
    fake_linear_quant_per_axis,
    fake_linear_quant_per_tensor,
    get_scale_zp_for_asym_quant,
)
from yoda.utils import torch_utils


def test_get_scale_zp_for_asym_quant():
    x_max = torch.tensor(0.5)
    x_min = torch.tensor(-1.8)
    scale, zp = get_scale_zp_for_asym_quant(x_min, x_max, 0, 255)
    torch.testing.assert_close(scale, torch.tensor(0.009020))
    torch.testing.assert_close(zp.float(), torch.tensor(200.0))


def test_fake_linear_quant_per_tensor():
    x = torch.rand(3, 4).requires_grad_()
    x_min = x.min()
    # here scale the maximum value by 0.5 to make some value stand out
    x_max = x.max() * 0.5
    s, zp = get_scale_zp_for_asym_quant(x_min, x_max, 0, 255)

    # forward
    out = fake_linear_quant_per_tensor(x, s, zp, 0, 255)
    expected = torch.fake_quantize_per_tensor_affine(
        x, s.item(), zp.long().item(), 0, 255
    )
    torch.testing.assert_close(out, expected)

    # backward
    out.backward(torch.ones_like(out))
    expected_grad = torch.where(
        torch.logical_and(x <= x_max, x >= x_min),
        torch.ones_like(x),
        torch.zeros_like(x),
    )
    torch.testing.assert_close(x.grad, expected_grad)


@pytest.mark.parametrize("axis", [1])
def test_fake_linear_quant_per_axis(axis):
    x = torch.rand(3, 4).requires_grad_()
    x_min = torch_utils.reduce_min(x, excluded_axis=axis, keepdim=True)
    x_max = torch_utils.reduce_max(x, excluded_axis=axis, keepdim=True) * 0.5
    s, zp = get_scale_zp_for_asym_quant(
        x_min.squeeze(), x_max.squeeze(), 0, 255
    )

    # forward
    out = fake_linear_quant_per_axis(
        x, s, zp, axis=1, min_bound=0, max_bound=255
    )
    expected = torch.fake_quantize_per_channel_affine(
        x, s, zp.float(), axis=1, quant_min=0, quant_max=255
    )
    torch.testing.assert_close(out, expected)

    # backward
    out.backward(torch.ones_like(out))
    expected_grad = torch.where(
        torch.logical_and(x <= x_max, x >= x_min),
        torch.ones_like(x),
        torch.zeros_like(x),
    )
    torch.testing.assert_close(x.grad, expected_grad)
