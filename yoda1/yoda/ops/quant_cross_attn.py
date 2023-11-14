import torch
from diffusers.models.attention import CrossAttention
from einops import rearrange
from loguru import logger
from torch_int.nn.bmm import BMM_S8T_S8N_F32T, BMM_S8T_S8N_S8T
from torch_int.nn.linear import W8A8B8O8Linear, W8A8BFP32OFP32Linear


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def is_2_power(n: int):
    return next_power_of_2(n) == n


def check_weight(m: torch.nn.Linear):
    weight = m.weight
    for s in weight.shape:
        # assert is_2_power(s), f'Invalid shape: {weight.shape}'
        pass


def add_bias_if_none(m: torch.nn.Linear):
    if m.bias is not None:
        return
    m.bias = torch.nn.Parameter(
        torch.zeros(size=(m.weight.shape[1],), dtype=m.weight.dtype).to(
            m.weight.device
        )
    )


class QuantCrossAttention(CrossAttention):
    enable_flash_attention = True
    concat_qkv_weight = True

    def __init__(self, *args, **kwargs):
        quant_kwargs = {}
        for name in ["input_scale", "out_input_scale"] + [
            f"{x}_output_scale" for x in "qkv"
        ]:
            quant_kwargs[name] = kwargs.pop(name)

        super().__init__(*args, **kwargs)

        # from fp32 model to quantization version
        logger.info("Initialize q_proj layer...")
        check_weight(self.to_q)
        add_bias_if_none(self.to_q)
        self.to_q = W8A8B8O8Linear.from_float(
            self.to_q,
            quant_kwargs["input_scale"],
            quant_kwargs["q_output_scale"],
        )
        logger.info("Initialize k_proj layer...")
        check_weight(self.to_k)
        add_bias_if_none(self.to_k)
        self.to_k = W8A8B8O8Linear.from_float(
            self.to_k,
            quant_kwargs["input_scale"],
            quant_kwargs["k_output_scale"],
        )
        logger.info("Initialize v_proj layer...")
        check_weight(self.to_v)
        add_bias_if_none(self.to_v)
        self.to_v = W8A8B8O8Linear.from_float(
            self.to_v,
            quant_kwargs["input_scale"],
            quant_kwargs["v_output_scale"],
        )

        logger.info("Initialize out_proj layer...")
        out_modules = []

        out_modules.append(
            W8A8BFP32OFP32Linear.from_float(
                self.to_out[0], quant_kwargs["out_input_scale"]
            )
        )
        # 使用int8输出，能狗降低latency到900us，但是仍然比不过flash attention版本
        # out_modules.append(
        #     W8A8B8O8Linear.from_float(
        #         self.to_out[0], quant_kwargs['out_input_scale'], 1.
        #     )
        # )
        out_modules.append(self.to_out[1])  # dropout
        self.to_out = torch.nn.ModuleList(out_modules)

        logger.info("Initialize attention layer...")
        self.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            quant_kwargs["q_output_scale"], quant_kwargs["k_output_scale"]
        )
        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        self.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127,
            quant_kwargs["v_output_scale"],
            quant_kwargs["out_input_scale"],
        )

    def forward(
        self, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        if attention_mask is not None:
            raise NotImplementedError("Attention mask not supported")
        if self.added_kv_proj_dim:
            raise NotImplementedError("add kv_proj dim not tested")

        if self.group_norm is not None:
            hidden_states = self.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        q = self.to_q(hidden_states)
        input_kv = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        k = self.to_k(input_kv)
        v = self.to_v(input_kv)

        attn_weights = self.qk_bmm(q, k)
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_probs.mul_(127).round_()
        attn_probs = attn_probs.to(torch.int8)
        out = self.pv_bmm(attn_probs, v)

        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out
