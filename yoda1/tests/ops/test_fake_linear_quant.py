import pytest
import torch

from yoda.ops.triton.quant import fake_linear_quant
from yoda.quant.linear_quant import get_scale_zp_for_asym_quant
from yoda.utils.torch_utils import reduce_max, reduce_min


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_triton_fake_linear_quant_per_tensor():
    x = torch.rand(3, 4).cuda()
    s, zp = get_scale_zp_for_asym_quant(x.min(), x.max() * 0.5, 0, 255)

    out = fake_linear_quant(x, s, zp, 0, 255)
    expected = (torch.round(x / s + zp).clamp_(0, 255) - zp) * s
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_triton_fake_linear_quant_per_axis():
    x = torch.rand(3, 4, 5, 6).cuda()

    axis = 1
    x_min = reduce_min(x, excluded_axis=axis, keepdim=True)
    x_max = reduce_max(x, excluded_axis=axis, keepdim=True)
    s, zp = get_scale_zp_for_asym_quant(x_min, x_max * 0.5, 0, 255)

    out = fake_linear_quant(x, s, zp, 0, 255, axis=axis)
    expected = (torch.round(x / s + zp).clamp_(0, 255) - zp) * s
    torch.testing.assert_close(out, expected)
