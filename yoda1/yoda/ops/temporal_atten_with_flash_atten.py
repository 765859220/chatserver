import flash_attn
import torch
from einops import rearrange
from libs.models.motion_module import VersatileAttention


class Rearrange(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return rearrange(x, *args, **kwargs)


class VersatileAttentionFlashAttn(VersatileAttention):
    enable_flash_attention = True

    def _forward(
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
            out = flash_attn.flash_attn_qkvpacked_func(qkv)
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            q = self.to_q(hidden_states)
            kv = torch.nn.functional.linear(encoder_hidden_states, self.to_kv)
            q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
            kv = rearrange(kv, "b n (t h d) -> b n t h d", t=2, h=self.heads)
            out = flash_attn.flash_attn_kvpacked_func(q, kv)
            out = rearrange(out, "b n h d -> b n (h d)")

        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
    ):
        if self.added_kv_proj_dim is not None:
            raise NotImplementedError
        if attention_mask is not None:
            raise NotImplementedError

        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )

            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)

            # encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        if self.group_norm is not None:
            hidden_states = self.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        out = self._forward(
            hidden_states, encoder_hidden_states, attention_mask
        )

        if self.attention_mode == "Temporal":
            out = rearrange(out, "(b d) f c -> (b f) d c", d=d)

        return out
