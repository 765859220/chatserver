import os

import pytest
import torch

import yoda

model_checkpoint = "/nas/wangxixi/yoda/assets/cross_attention/cross_atten.pth"


@pytest.mark.skipif(
    not os.path.isfile(model_checkpoint),
    reason=f"checkpoint not found: {model_checkpoint}",
)
@torch.no_grad()
def test_cross_attention_with_flash_attention():
    # load data
    inputs = {}
    inputs["encoder_hidden_states"] = (
        (torch.rand(size=(16, 77, 768)) * 2 - 1).cuda().half()
    )
    inputs["hidden_states"] = (
        (torch.rand(size=(16, 2560, 320)) * 2 - 1).cuda().half()
    )

    import diffusers
    from diffusers.models.attention import CrossAttention

    ori: CrossAttention = torch.load(model_checkpoint).eval()

    from yoda.ops.cross_attn_with_flash_atten import CrossAttentionFlashAttn

    setattr(
        diffusers.models.attention, "CrossAttention", CrossAttentionFlashAttn
    )
    module: CrossAttentionFlashAttn = torch.load(model_checkpoint).eval()

    expected = ori(**inputs)
    out = module(**inputs)

    torch.testing.assert_close(out, expected=expected, atol=1e-3, rtol=1)
