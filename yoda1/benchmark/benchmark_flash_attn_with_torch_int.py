import torch

import yoda

model_checkpoint = "/nas/wangxixi/yoda/assets/cross_attention/cross_atten.pth"


def load_data():
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
    return inputs, quant_inputs


def load_naive_model():

    from diffusers.models.attention import CrossAttention

    fp32_module: CrossAttention = torch.load(model_checkpoint).cuda().eval()
    return fp32_module


def load_flash_attn_fp16_model():
    import diffusers

    from yoda.ops.cross_attn_with_flash_atten import CrossAttentionFlashAttn

    setattr(
        diffusers.models.attention, "CrossAttention", CrossAttentionFlashAttn
    )
    module: CrossAttentionFlashAttn = torch.load(model_checkpoint).eval()
    return module


def load_flash_attn_triton_fp16_model():
    import diffusers

    from yoda.ops.cross_attn_with_flash_atten import (
        CrossAttentionFlashAttnTriton,
    )

    setattr(
        diffusers.models.attention,
        "CrossAttention",
        CrossAttentionFlashAttnTriton,
    )
    module: CrossAttentionFlashAttnTriton = torch.load(model_checkpoint).eval()
    return module


def load_quant_attention_model(fp32_module):
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
    return quant_module.cuda()


@torch.no_grad()
def run_deepspeed_benchmark():
    print("load data...")
    input_dict, quant_input_dict = load_data()
    print("load naive model...")
    naive_model = load_naive_model()
    quant_model = load_quant_attention_model(naive_model)
    flash_attn_model = load_flash_attn_fp16_model()
    flash_attn_triton_model = load_flash_attn_triton_fp16_model()

    from yoda.profiler.profiler import get_model_profile

    get_model_profile(
        naive_model, kwargs=input_dict, print_profile=True, detailed=True
    )
    get_model_profile(
        flash_attn_model, kwargs=input_dict, print_profile=True, detailed=True
    )
    get_model_profile(
        flash_attn_triton_model,
        kwargs=input_dict,
        print_profile=True,
        detailed=True,
    )
    get_model_profile(
        quant_model, kwargs=quant_input_dict, print_profile=True, detailed=True
    )


if __name__ == "__main__":
    run_deepspeed_benchmark()
