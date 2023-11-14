import sys
from typing import List, Optional

import torch
import typer
from loguru import logger

from yoda.viz.viz import visualize

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
    save_prefix: str,
    depth: Optional[List[int]] = typer.Option(None),
):
    if not depth:
        module_depth = [0]
    else:
        module_depth = depth

    data = get_input_dict()

    from bench_text2video import load_model

    model = load_model()
    net = model.unet.cuda().half()

    for depth in module_depth:
        logger.info(f"Visualize model at depth level: {depth}")
        filename = save_prefix + f"_depth_{depth}"
        visualize(net, input_data=data, depth=depth, save_path=filename)
        logger.info(f"Check saved image: {filename}.png")


if __name__ == "__main__":
    typer.run(main)
