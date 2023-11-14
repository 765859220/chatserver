import torch
from diffusers.models.attention import CrossAttention
from einops import rearrange
from flash_attn import flash_attn_kvpacked_func, flash_attn_qkvpacked_func

from yoda.ops.triton.flash_attention import attention


class CrossAttentionFlashAttn(CrossAttention):
    def forward(
        self, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        if attention_mask is not None:
            raise NotImplementedError
        if self.added_kv_proj_dim:
            assert False, "not tested"

        if self.group_norm is not None:
            hidden_states = self.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        if not hasattr(self, "to_qkv") or hasattr(self, "to_kv"):
            if encoder_hidden_states is None:
                self.to_qkv = torch.cat(
                    (self.to_q.weight, self.to_k.weight, self.to_v.weight)
                )
            else:
                self.to_kv = torch.cat((self.to_k.weight, self.to_v.weight))

        if hasattr(self, "to_qkv"):
            qkv = torch.nn.functional.linear(hidden_states, self.to_qkv)
            qkv = rearrange(qkv, "b n (t h d) -> b n t h d", h=self.heads, t=3)
            out = flash_attn_qkvpacked_func(qkv)
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            q = self.to_q(hidden_states)
            kv = torch.nn.functional.linear(encoder_hidden_states, self.to_kv)
            q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
            kv = rearrange(kv, "b n (t h d) -> b n t h d", t=2, h=self.heads)
            out = flash_attn_kvpacked_func(q, kv)
            out = rearrange(out, "b n h d -> b n (h d)")

        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class CrossAttentionFlashAttnTriton(CrossAttentionFlashAttn):
    def forward(
        self, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        if attention_mask is not None:
            raise NotImplementedError
        if self.added_kv_proj_dim:
            assert False, "not tested"

        if self.group_norm is not None:
            hidden_states = self.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        q = self.to_q(hidden_states)
        input_state = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        k = self.to_k(input_state)
        v = self.to_v(input_state)
        q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.heads)
        out = attention(q, k, v, False, 0.5)

        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out
