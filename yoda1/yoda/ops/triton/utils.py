from typing import Callable, List

import triton


def make_benchmark(title, x_names, x_vals, args, **kwargs):
    benchmark_info = {
        "x_names": x_names,
        "x_vals": x_vals,
        "line_arg": "backend",
        "line_vals": ["torch", "triton"],
        "line_names": ["torch", "triton"],
        "plot_name": title,
        "args": args,
        "ylabel": "milliseconds",
    }
    benchmark_info.update(**kwargs)
    return triton.testing.Benchmark(**benchmark_info)


def report(title, x_names, x_vals, args=None, **kwargs):
    if args is None:
        args = {}
    return triton.testing.perf_report(
        [make_benchmark(title, x_names, x_vals, args, **kwargs)]
    )


def run_benchmark(func, title, x_names, x_vals, args=None, **kwargs):
    bench = report(title, x_names, x_vals, args, **kwargs)(func)
    bench.run(show_plots=False, print_data=True, save_path="./benchmark")


def autotune(
    configs: List[triton.Config],
    key: List[str],
    prune_configs_by: Callable = None,
    reset_to_zero: List[str] = None,
    warmup: int = 25,
    rep: int = 100,
):
    return triton.autotune(
        configs, key, prune_configs_by, reset_to_zero, warmup, rep
    )
