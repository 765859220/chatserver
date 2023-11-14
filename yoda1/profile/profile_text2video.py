import sys
from typing import Optional

import torch
import typer
from loguru import logger

from yoda.ops import build_model_with_yoda_ops
from yoda.profiler.profiler import get_model_profile

sys.path.append("/vepfs/home/wangxixi/sd_benchmark/3rdparty/videodiffusion")
sys.path.append("/vepfs/home/wangxixi/sd_benchmark/")


def get_input_dict():
    num_frame = 16
    batch_size = 8

    sample = torch.rand(batch_size, 4, num_frame, 40, 64).cuda().half()
    t = torch.tensor(999).long().cuda().half()
    text_emb = torch.rand(batch_size, 77, 768).cuda().half()

    return {"sample": sample, "timestep": t, "encoder_hidden_states": text_emb}


def main(
    out_file: Optional[str] = None,
    use_flash_atten: bool = False,
    use_temporal_atten: bool = False,
    use_xformers: bool = False,
):
    if use_xformers:
        assert not use_flash_atten and not use_temporal_atten

    # custom leaf modules to benchmark
    from diffusers.models.attention import CrossAttention
    from libs.models.motion_module import VersatileAttention

    custom_leaf_modules = [CrossAttention, VersatileAttention]

    if use_flash_atten:
        from yoda.ops.cross_attn_with_flash_atten import CrossAttentionFlashAttn

        custom_leaf_modules.append(CrossAttentionFlashAttn)

    if use_temporal_atten:
        from yoda.ops.temporal_atten_with_flash_atten import (
            VersatileAttentionFlashAttn,
        )

        custom_leaf_modules.append(VersatileAttentionFlashAttn)

    # build model with yoda ops
    logger.info("Try to enable yoda ops...")
    with build_model_with_yoda_ops(
        enable_flash_atten_in_cross_atten=use_flash_atten,
        enable_flash_atten_in_temporal_atten=use_temporal_atten,
    ) as manager:
        from bench_text2video import load_model

        model = load_model()
        unet = model.unet.to(torch.device("cuda:0"), dtype=torch.float16)

        manager.check(unet)

    if use_xformers:
        logger.info("Try to enable xformers optimization...")
        from libs.models.openaimodel_nvidia import UNet3DConditionModel

        unet: UNet3DConditionModel
        unet.enable_xformers_memory_efficient_attention()

        for name, module in unet.named_modules():
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                logger.info(
                    f"xformer accelerated module: {name}, type: {type(module)}"
                )

    # leaf module will not be step into to profile
    def is_leaf_module(m, quan_name):
        if isinstance(m, torch.nn.Sequential):
            return False
        if m.__module__.startswith("torch.nn") or m.__module__.startswith(
            "torch.ao.nn"
        ):
            return True
        if next(m.named_children(), None) is None:
            return True
        if type(m) in custom_leaf_modules:
            return True
        return False

    # let's go
    flops, macs, params = get_model_profile(
        unet,
        kwargs=get_input_dict(),
        print_profile=True,
        detailed=True,
        top_modules=3,
        output_file=out_file,
        is_leaf_module=is_leaf_module,
    )
    if out_file is not None:
        with open(out_file, "r") as f:
            print(f.read())


if __name__ == "__main__":
    typer.run(main)
