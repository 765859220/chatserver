from itertools import product

import torch
import triton

from yoda.ops.triton.quant import fake_linear_quant
from yoda.ops.triton.utils import run_benchmark
from yoda.quant.linear_quant import (
    fake_linear_quant_per_axis,
    fake_linear_quant_per_tensor,
)


def benchmark_fake_linear_quant_per_tensor(size, backend):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    s = torch.rand(size=(1,), device="cuda", dtype=torch.float32)
    zp = torch.tensor(0, device="cuda", dtype=torch.int64)

    quantiles = [0.5, 0.1, 0.9]
    if backend == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fake_linear_quant_per_tensor(x, s, zp, 0, 255),
            quantiles=quantiles,
        )
    if backend == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fake_linear_quant(x, s, zp, 0, 255), quantiles=quantiles
        )
    return ms, max_ms, min_ms


# run_benchmark(
#     benchmark_fake_linear_quant_per_tensor,
#     title="benchmark_fake_linear_quant_per_tensor",
#     x_names=["size"],
#     x_vals=[2**i for i in range(10, 30)],
#     x_log=True,
# )


def benchmark_fake_linear_quant_per_axis(channel, height, backend):
    x = torch.rand(
        size=(128, channel, height, height), device="cuda", dtype=torch.float32
    )
    s = torch.rand(size=(channel,), device="cuda", dtype=torch.float32)
    zp = torch.zeros_like(s).long()

    quantiles = [0.5, 0.1, 0.9]
    if backend == "torch":
        s = s.reshape(1, channel, 1, 1)
        zp = zp.reshape(1, channel, 1, 1)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fake_linear_quant_per_axis(
                x, s, zp, axis=1, min_bound=0, max_bound=255
            ),
            quantiles=quantiles,
        )
    if backend == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fake_linear_quant(x, s, zp, 0, 255, axis=1),
            quantiles=quantiles,
        )
    return ms, max_ms, min_ms


run_benchmark(
    benchmark_fake_linear_quant_per_axis,
    title="benchmark_fake_linear_quant_per_axis",
    x_names=["channel", "height"],
    x_vals=product([4, 8, 16, 32, 64, 128, 256, 512], [32]),
)
