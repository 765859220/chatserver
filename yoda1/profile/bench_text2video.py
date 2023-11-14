"""
python bench_sd.py /nas/snow/stable-diffusion-v1-4 sd_v14_compile --num-warmup 3 --num-repeat 10
"""
# flake8: noqa
import os
from typing import List

import sys
from ipdb import set_trace
sys.path.append("/vepfs/home/chendong/sd_benchmark1/3rdparty/videodiffusion")
sys.path.append("/vepfs/home/chendong/sd_benchmark1/")

# sys.path.append("/vepfs/home/wangxixi/sd_benchmark/3rdparty/videodiffusion")
# sys.path.append("/vepfs/home/wangxixi/sd_benchmark/")

import diffusers
import numpy as np
import torch
import torch.nn as nn
import transformers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from libs.models.openaimodel_nvidia import UNet3DConditionModel
from libs.pipelines.pipeline_generatekeyframes import GenerateKeyFramesPipeline
from libs.util import ddim_inversion, save_images_grid, save_videos_grid
from libs.utils.convert_from_ckpt import (
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
)
from libs.utils.convert_lora_safetensor_to_diffusers import convert_lora
from omegaconf import OmegaConf
from safetensors import safe_open
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

transformers.logging.set_verbosity_error()
import json
import os
import time

import torch
import typer
from accelerate.utils import set_seed
from loguru import logger
from omegaconf import OmegaConf


def get_device(m):
    return next(m.parameters()).device


def load_model():
    seednum = 0
    do_motion_our = True
    if not do_motion_our:
        test_with_mm = True  # do not change
        replace_sd = True  # do not change
        use_linear = True  # do not change
        use_motion_new = True  # do not change
        use_temp_conv = False  # do not change
        prefixname = "mm08"
    else:
        ## our param
        test_with_mm = False  # do not #change
        replace_sd = True  # do not change # 0.3.0及以后设置为True
        use_linear = (
            True  # may change respect to train scheduler #0.2.2 及以后的版本设置为True
        )
        use_motion_new = (
            True  # may change respect to model arch #0.2.2 及以后的版本设置为True
        )
        use_temp_conv = True
        prefixname = "our"  #
    temporal_position_encoding_max_len = 24
    Height = 320
    Width = 512
    video_length = 16
    video_save_path = "{}_text_to_video_{}{}{}_{}.gif".format(
        prefixname, Height, Width, video_length, seednum
    )

    pretrained_model_path = "/nas/snow/stable-diffusion-v1-5/"  # "/nas/snow/sd_1_5_new_dd" #stable-diffusion-v1-5" #stable-diffusion-v1-5/"
    inference_config = OmegaConf.load(
        "/nas/snow/code/AnimateDiff/configs/inference/inference.yaml"
    )
    animate_mm_path = "/nas/snow/checkpoint_8000.ckpt"  # "/nas/snow/mm_sd_v15.ckpt" ## 45000, 39000

    pretrained_model_path = "/nas/snow/stable-diffusion-v1-5/"  # "/nas/snow/sd_1_5_new_dd" #stable-diffusion-v1-5" #stable-diffusion-v1-5/"
    inference_config = OmegaConf.load(
        "/nas/snow/code/AnimateDiff/configs/inference/inference.yaml"
    )
    animate_mm_path = "/nas/snow/checkpoint_8000.ckpt"  # "/nas/snow/mm_sd_v15.ckpt" ## 45000, 39000
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path,
        subfolder="unet",
        use_motion_module=use_motion_new,
        use_temp_conv=use_temp_conv,
        temporal_position_encoding_max_len=temporal_position_encoding_max_len,
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # DPMSolverMultistepScheduler
    scheduler_linear = DPMSolverMultistepScheduler(
        **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
    )  # DDIMScheduler
    scheduler_scale = DPMSolverMultistepScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler"
    )  # DDIMScheduler

    pipeline = GenerateKeyFramesPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler_linear if use_linear else scheduler_scale,
    )

    pipeline.enable_vae_slicing()

    our_mm_path = "/nas/snow/videoexp/a100_16frames_stride16_ft_sd15new_7machine_webvid/output1e-4_fs16_motionnew_origrad_cov_alltrain/checkpoint-10000/pytorch_model.bin"
    model_config_path = "/nas/snow/epicrealism_pureEvolutionV5.safetensors"  ##replace_model_path
    model_config_base = ""  ##replace_sd path

    if test_with_mm:
        motion_module_state_dict = torch.load(
            animate_mm_path, map_location="cpu"
        )
        if "global_step" in motion_module_state_dict:
            # func_args.update({"global_step": motion_module_state_dict["global_step"]})
            motion_module_state_dict = motion_module_state_dict["state_dict"]
        convert_dict = {}
        for key in motion_module_state_dict.keys():
            newkey = key.replace("motion_modules", "temp_attentions")
            newkey = newkey.replace("module.", "")
            convert_dict[newkey] = motion_module_state_dict[key]
        missing, unexpected = pipeline.unet.load_state_dict(
            convert_dict, strict=False
        )
        assert len(unexpected) == 0
        logger.info("load mm motion module: {}".format(animate_mm_path))
    else:
        # import pdb;pdb.set_trace()
        our_motion_module_state_dict = torch.load(
            our_mm_path, map_location="cpu"
        )
        missing, unexpected = pipeline.unet.load_state_dict(
            our_motion_module_state_dict, strict=True
        )
        assert len(our_motion_module_state_dict.keys()) == len(
            pipeline.unet.state_dict().keys()
        )
        logger.info("load our motion module: {}".format(our_mm_path))

    if replace_sd:
        if model_config_path != "":
            print("load replace sd: {}".format(model_config_path))
            if model_config_path.endswith(".ckpt"):
                state_dict = torch.load(model_config_path)
                pipeline.unet.load_state_dict(state_dict)

            elif model_config_path.endswith(".safetensors"):
                state_dict = {}
                with safe_open(
                    model_config_path, framework="pt", device="cpu"
                ) as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

                is_lora = all("lora" in k for k in state_dict.keys())
                if not is_lora:
                    base_state_dict = state_dict
                else:
                    base_state_dict = {}
                    with safe_open(
                        model_config_base, framework="pt", device="cpu"
                    ) as f:
                        for key in f.keys():
                            base_state_dict[key] = f.get_tensor(key)

                # vae
                converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                    base_state_dict, pipeline.vae.config
                )
                pipeline.vae.load_state_dict(converted_vae_checkpoint)
                print("load replace sd vae")
                # unet
                converted_unet_checkpoint = convert_ldm_unet_checkpoint(
                    base_state_dict, pipeline.unet.config
                )
                pipeline.unet.load_state_dict(
                    converted_unet_checkpoint, strict=False
                )
                print("load replace sd unet")
                # text_model
                pipeline.text_encoder = convert_ldm_clip_checkpoint(
                    base_state_dict
                )
                print("load replace sd text encoder")
    return pipeline


def get_pxx(values: List[float], p: float) -> float:
    assert 0 <= p <= 1
    values.sort()
    return values[int(len(values) * p)]


def prepare(
    pipe,
    use_compile: bool,
    use_xformers: bool,
    use_deepspeed: bool,
    compile_mode: str = "reduce-overhead",
):
    device = torch.device("cuda:0")
    pipe = pipe.to(device)
    pipe.text_encoder.to(device, dtype=torch.float32)
    pipe.vae.to(device, dtype=torch.float32)
    pipe.unet.to(device, dtype=torch.float32)

    # ait
    # pipe.unet.eval()
    # from fx2ait.acc_tracer import acc_tracer

    # xt = torch.rand(1, 4, 16, 320, 512)
    # t = torch.rand(1,)
    # text_emb = torch.rand(1, 77, 768)

    # mod = acc_tracer.trace(pipe.unet, [xt, t, text_emb])
    # print(mod)

    # from ipdb import set_trace
    # set_trace()

    if use_compile:
        # currently, not work in A800
        logger.info(f"torch.compile with mode: {compile_mode}")
        pipe.unet = torch.compile(pipe.unet, mode=compile_mode)
    else:
        logger.info("torch.compile not enabled")
    if use_xformers:
        if is_xformers_available():
            logger.info(f"Enable xformers")
            pipe.unet.enable_xformers_memory_efficient_attention()
        else:
            raise RuntimeError("xformer not installed")
    else:
        logger.info("xformers not enabled")

    if use_deepspeed:
        import deepspeed

        pipe = deepspeed.init_inference(
            pipe,
            mp_size=1,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            enable_cuda_graph=True,
        )
    else:
        logger.info("deepspeed not enabled")

    return pipe


def infer(
    pipe,
    outpath,
    prompt_length: int = 32,
):
    seednum = 33
    set_seed(seednum)
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(seednum)

    # prompt = 'x' * prompt_length
    prompt_list = ["a dog playing with a girl",
              "A small bird sits atop a blooming flower stem.",
              "doctors are constructing a robot",
              "a young girl playing piano, slow motion"]
    Height = 320
    Width = 512
    video_length = 16
    for i, prompt in enumerate(prompt_list):
        sample = pipe(
            prompt,
            num_inference_steps=25,
            generator=generator,
            latents=None,
            height=Height,
            width=Width,
            video_length=video_length,
            num_videos_per_prompt=1,
            guidance_scale=7.5,
        ).videos
        cur_video_save_path = f'{outpath}/{i}_full.gif'
        save_videos_grid(sample, cur_video_save_path)
    return


def main(
    bench_name: str,
    num_warmup: int = 3,
    num_repeat: int = 10,
    len_prompt: int = 32,
    use_compile: bool = False,
    use_xformers: bool = False,
    use_deepspeed: bool = False,
    compile_mode: str = "reduce-overhead",
    output_file: str = "output/benchmark.json",
):
    """the main entrypoint for benchmark, write the result into OUTPUT_FILE"""
    # pipe = load_model()
    # logger.info("Prepare the model (magic happens here)")
    import dill
    with open('/vepfs/home/chendong/code/q-diffusion/quant_all.pkl', 'rb') as f:
        pipe = dill.load(f)
    logger.info("Prepare the model from pkl (magic happens here)")
    pipe = prepare(
        pipe,
        use_compile=use_compile,
        use_xformers=use_xformers,
        use_deepspeed=use_deepspeed,
        compile_mode=compile_mode,
    )

    # warmup
    logger.info(f"Run warmup {num_warmup} times")
    for _ in range(num_warmup):
        infer(pipe, len_prompt)


if __name__ == "__main__":
    typer.run(main)
