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
def test_quant_cross_attention_could_run():
    # load data
    inputs = {}
    inputs["encoder_hidden_states"] = (
        (torch.rand(size=(16, 80, 768)) * 2 - 1).cuda().half()
    )
    inputs["hidden_states"] = (
        (torch.rand(size=(16, 2560, 320)) * 2 - 1).cuda().half()
    )

    quant_inputs = {}
    for name, value in inputs.items():
        quant_inputs[name] = (value * 127).round().to(torch.int8)

    from diffusers.models.attention import CrossAttention

    fp32_module: CrossAttention = torch.load(model_checkpoint).eval()

    from yoda.ops.quant_cross_attn import QuantCrossAttention

    quant_module = QuantCrossAttention(
        query_dim=fp32_module.to_q.weight.shape[0],
        cross_attention_dim=fp32_module.to_k.weight.shape[0],
        heads=fp32_module.heads,
        dim_head=fp32_module.to_q.weight.shape[1] // fp32_module.heads,
        dropout=0.0,
        bias=fp32_module.to_q.bias is not None,
        upcast_attention=fp32_module.upcast_attention,
        upcast_softmax=fp32_module.upcast_softmax,
        added_kv_proj_dim=fp32_module.added_kv_proj_dim,
        norm_num_groups=None,
        # quant config
        input_scale=1.0 / 127,
        out_input_scale=1.0,
        q_output_scale=1.0,
        k_output_scale=1.0,
        v_output_scale=1.0,
    )

    quant_module = quant_module.to("cuda")
    _ = quant_module(**quant_inputs)
