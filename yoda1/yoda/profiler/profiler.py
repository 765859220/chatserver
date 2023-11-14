""" a profiler based on deepspeed """
import inspect
from collections import defaultdict
from typing import Callable, Dict

import einops
import numpy as np
import torch
from loguru import logger
from tabulate import tabulate
from torch import nn

from yoda.utils.misc import number_to_string


def _upsample_flops_compute(*args, **kwargs):
    input = args[0]
    size = kwargs.get("size", None)
    if size is None and len(args) > 1:
        size = args[1]

    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            return int(np.prod(size)), 0
        else:
            return int(size), 0

    scale_factor = kwargs.get("scale_factor", None)
    if scale_factor is None and len(args) > 2:
        scale_factor = args[2]
    assert (
        scale_factor is not None
    ), "either size or scale_factor should be defined"

    flops = input.numel()
    if isinstance(scale_factor, (list, tuple)):
        assert len(scale_factor) == input.ndim - 2
        flops * int(np.prod(scale_factor))
    else:
        flops * scale_factor ** len(input)
    return flops, 0


import deepspeed  # noqa: E402

setattr(
    deepspeed.profiling.flops_profiler.profiler,
    "_upsample_flops_compute",
    _upsample_flops_compute,
)

from deepspeed.profiling.flops_profiler.profiler import (  # noqa: E402
    BACKWARD_GLOBAL_TIMER,
    DEFAULT_PRECISION,
    FORWARD_GLOBAL_TIMER,
    STEP_GLOBAL_TIMER,
    FlopsProfiler,
    duration_to_string,
    flops_to_string,
    get_module_duration,
    get_module_flops,
    get_module_macs,
    macs_to_string,
    params_to_string,
    wrapFunc,
)


def _rearrange_flops_compute(input, *args, **kwargs):
    flops = input.numel()
    return flops, 0


def is_leaf_module(m, quan_name):
    if isinstance(m, torch.nn.Sequential):
        return False
    if m.__module__.startswith("torch.nn") or m.__module__.startswith(
        "torch.ao.nn"
    ):
        return True
    if next(m.named_children(), None) is None:
        return True
    return False


class SummaryInfo:
    def __init__(self):
        self.duration = {}
        self.mac = {}
        self.flops = {}
        self.is_leaf = False

    def update(self, module: nn.Module, name: str, is_leaf: bool):
        self.flops[name] = get_module_flops(module)
        self.mac[name] = get_module_macs(module)
        self.duration[name] = get_module_duration(module)
        self.is_leaf = is_leaf

    def get_names(self):
        return list(self.duration.keys())

    def total_duration(self):
        return sum(self.duration.values())

    def get_duration_stat(self):
        durations = list(self.duration.values())
        return np.mean(durations), np.min(durations), np.max(durations)


class YodaProfiler(FlopsProfiler):
    """Measures the latency, number of estimated floating-point operations and parameters of each module in a PyTorch model.

    The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how latency, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top k modules in terms of aggregated latency, flops, and parameters at depth l with k and l specified by the user. The output profile is computed for each batch of input.
    The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package.
    When using DeepSpeed for model training, the flops profiler can be configured in the deepspeed_config file and no user code change is required.

    If using the profiler as a standalone package, one imports the flops_profiler package and use the APIs.

    Here is an example for usage in a typical training workflow:

        .. code-block:: python

            model = Model()
            prof = FlopsProfiler(model)

            for step, batch in enumerate(data_loader):
                if step == profile_step:
                    prof.start_profile()

                loss = model(batch)

                if step == profile_step:
                    flops = prof.get_total_flops(as_string=True)
                    params = prof.get_total_params(as_string=True)
                    prof.print_model_profile(profile_step=profile_step)
                    prof.end_profile()

                loss.backward()
                optimizer.step()

    To profile a trained model in inference, use the `get_model_profile` API.

    Args:
        object (torch.nn.Module): The PyTorch model to profile.
    """

    def __init__(self, model, ds_engine=None, recompute_fwd_factor=0.0):
        super().__init__(model, ds_engine, recompute_fwd_factor)
        # how to compute flops of given functionals
        self.custom_patch_functionals: Dict[Callable, Callable] = {
            einops.rearrange: _rearrange_flops_compute,
        }

    def start_profile(self, ignore_list=None):
        # enable custom patch functionals
        for ori_fun, flops_calculator in self.custom_patch_functionals.items():
            logger.info(f"Wrapfunc for {ori_fun}")
            module = inspect.getmodule(ori_fun)
            name = ori_fun.__name__
            setattr(module, name, wrapFunc(ori_fun, flops_calculator))

        return super().start_profile(ignore_list)

    def stop_profile(self):
        # remove custom patch functionals
        if self.started and self.func_patched:
            for ori_fun in self.custom_patch_functionals:
                logger.info(f"Restore func for {ori_fun}")
                module = inspect.getmodule(ori_fun)
                name = ori_fun.__name__
                setattr(module, name, ori_fun)

        super().stop_profile()

    def print_model_profile(
        self,
        profile_step=1,
        module_depth=-1,
        top_modules=1,
        detailed=True,
        output_file=None,
        is_leaf_module=None,
    ):
        """Prints the model graph with the measured profile attached to each module.

        Args:
            profile_step (int, optional): The global training step at which to profile. Note that warm up steps are needed for accurate time measurement.
            module_depth (int, optional): The depth of the model to which to print the aggregated module information. When set to -1, it prints information from the top to the innermost modules (the maximum depth).
            top_modules (int, optional): Limits the aggregated profile output to the number of top modules specified.
            detailed (bool, optional): Whether to print the detailed model profile.
            output_file (str, optional): Path to the output file. If None, the profiler prints to stdout.
        """
        if not self.started:
            return
        import os.path
        import sys

        original_stdout = None
        f = None
        if output_file:
            dir_path = os.path.dirname(os.path.abspath(output_file))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            original_stdout = sys.stdout
            f = open(output_file, "w")
            sys.stdout = f

        total_flops = self.get_total_flops()
        total_macs = self.get_total_macs()
        total_duration = self.get_total_duration()
        total_params = self.get_total_params()
        expert_tensor_parallelism = None  # silence the linters
        total_model_expert_params = total_model_nonexpert_params = 0
        if self.ds_engine:
            total_model_nonexpert_params = (
                self.model.__params__ * self.ds_engine.mp_world_size
            )
            if self.ds_engine.has_moe_layers:
                expert_tensor_parallelism = (
                    self.ds_engine.mp_world_size
                    if self.is_expert_tensor_parallelism_enabled()
                    else 1
                )
                total_model_expert_params = (
                    self.model.__model_expert_params__
                    * expert_tensor_parallelism
                )

        self.flops = total_flops
        self.macs = total_macs
        self.params = total_params

        print(
            "\n-------------------------- DeepSpeed Flops Profiler --------------------------"
        )
        print(f"Profile Summary at step {profile_step}:")
        print(
            "Notations:\n"
            "data parallel size (dp_size), model parallel size(mp_size),\n"
            "number of parameters (params), number of multiply-accumulate operations(MACs),\n"
            "number of floating-point operations (flops), floating-point operations per second (FLOPS),\n"
            "fwd latency (forward propagation latency), bwd latency (backward propagation latency),\n"
            "step (weights update latency), iter latency (sum of fwd, bwd and step latency)\n"
        )
        line_fmt = "{:<70}  {:<8}"
        if self.ds_engine:
            print(line_fmt.format("world size: ", self.ds_engine.world_size))
            print(
                line_fmt.format(
                    "data parallel size: ", self.ds_engine.dp_world_size
                )
            )
            print(
                line_fmt.format(
                    "model parallel size: ", self.ds_engine.mp_world_size
                )
            )
            print(
                line_fmt.format(
                    "batch size per GPU: ",
                    self.ds_engine.train_micro_batch_size_per_gpu(),
                )
            )
            if self.ds_engine.has_moe_layers:
                print(
                    line_fmt.format(
                        "expert tensor parallelism enabled: ",
                        expert_tensor_parallelism > 1,
                    )
                )

        print(
            line_fmt.format("params per GPU: ", params_to_string(total_params))
        )
        if total_model_expert_params > 0:
            print(
                line_fmt.format(
                    "params of model: ",
                    params_to_string(
                        total_model_nonexpert_params + total_model_expert_params
                    ),
                )
            )
            print(
                line_fmt.format(
                    "   non-expert params of model: ",
                    params_to_string(total_model_nonexpert_params),
                )
            )
            print(
                line_fmt.format(
                    "   expert params of model: ",
                    params_to_string(total_model_expert_params),
                )
            )
        else:
            print(
                line_fmt.format(
                    "params of model = params per GPU * mp_size: ",
                    params_to_string(total_model_nonexpert_params),
                )
            )

        print(line_fmt.format("fwd MACs per GPU: ", macs_to_string(total_macs)))

        print(
            line_fmt.format(
                "fwd flops per GPU: ", number_to_string(total_flops)
            )
        )

        print(
            line_fmt.format(
                "fwd flops of model = fwd flops per GPU * mp_size: ",
                number_to_string(
                    total_flops
                    * (self.ds_engine.mp_world_size if self.ds_engine else 1)
                ),
            )
        )

        fwd_latency = self.get_total_duration()
        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            fwd_latency = (
                self.ds_engine.timers(FORWARD_GLOBAL_TIMER).elapsed(False)
                / 1000.0
            )
        print(line_fmt.format("fwd latency: ", duration_to_string(fwd_latency)))
        print(
            line_fmt.format(
                "fwd FLOPS per GPU = fwd flops per GPU / fwd latency: ",
                flops_to_string(total_flops / fwd_latency),
            )
        )

        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            bwd_factor = 2 + self.recompute_fwd_factor
            bwd_latency = (
                self.ds_engine.timers(BACKWARD_GLOBAL_TIMER).elapsed(False)
                / 1000.0
            )
            step_latency = (
                self.ds_engine.timers(STEP_GLOBAL_TIMER).elapsed(False) / 1000.0
            )
            print(
                line_fmt.format(
                    "bwd latency: ", duration_to_string(bwd_latency)
                )
            )
            print(
                line_fmt.format(
                    f"bwd FLOPS per GPU = {bwd_factor:g} * fwd flops per GPU / bwd latency: ",
                    flops_to_string(bwd_factor * total_flops / bwd_latency),
                )
            )
            print(
                line_fmt.format(
                    f"fwd+bwd FLOPS per GPU = {bwd_factor + 1:g} * fwd flops per GPU / (fwd+bwd latency): ",
                    flops_to_string(
                        (bwd_factor + 1)
                        * total_flops
                        / (fwd_latency + bwd_latency)
                    ),
                )
            )

            print(
                line_fmt.format(
                    "step latency: ", duration_to_string(step_latency)
                )
            )

            iter_latency = fwd_latency + bwd_latency + step_latency
            print(
                line_fmt.format(
                    "iter latency: ", duration_to_string(iter_latency)
                )
            )
            print(
                line_fmt.format(
                    f"FLOPS per GPU = {bwd_factor + 1:g} * fwd flops per GPU / iter latency: ",
                    flops_to_string(
                        (bwd_factor + 1) * total_flops / iter_latency
                    ),
                )
            )

            samples_per_iter = (
                self.ds_engine.train_micro_batch_size_per_gpu()
                * self.ds_engine.world_size
            )
            print(
                line_fmt.format(
                    "samples/second: ",
                    round(samples_per_iter / iter_latency, DEFAULT_PRECISION),
                )
            )

        def flops_repr(module):
            params = module.__params__ + module.__expert_params__
            flops = get_module_flops(module)
            macs = get_module_macs(module)
            duration = get_module_duration(module)
            items = [
                "{} = {:g}% Params".format(
                    params_to_string(params),
                    round(100 * params / total_params, DEFAULT_PRECISION)
                    if total_params
                    else 0,
                ),
                "{} = {:g}% MACs".format(
                    macs_to_string(macs),
                    round(100 * macs / total_macs, DEFAULT_PRECISION)
                    if total_macs
                    else 0,
                ),
                "{} = {:g}% latency".format(
                    duration_to_string(duration),
                    round(100 * duration / total_duration, DEFAULT_PRECISION)
                    if total_duration
                    else 0,
                ),
                flops_to_string(
                    round(flops / duration, DEFAULT_PRECISION)
                    if duration
                    else 0
                ),
            ]
            original_extra_repr = module.original_extra_repr()
            if original_extra_repr:
                items.append(original_extra_repr)
            return ", ".join(items)

        def add_extra_repr(module):
            flops_extra_repr = flops_repr.__get__(module)
            if module.extra_repr != flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = flops_extra_repr
                assert module.extra_repr != module.original_extra_repr

        def del_extra_repr(module):
            if hasattr(module, "original_extra_repr"):
                module.extra_repr = module.original_extra_repr
                del module.original_extra_repr

        self.model.apply(add_extra_repr)

        print(
            "\n----------------------------- Aggregated Profile per GPU -----------------------------"
        )
        self.print_model_aggregated_profile(
            module_depth=module_depth, top_modules=top_modules
        )

        if detailed:
            print(
                "\n------------------------------ Detailed Profile per GPU ------------------------------"
            )
            print(
                "Each module profile is listed after its name in the following order: \nparams, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS"
            )
            print(
                "\nNote: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.\n2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.\n3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.\n"
            )
            print(self.model)

        # summary

        counter = defaultdict(SummaryInfo)

        # def travel(m: nn.Module):
        #     for name, subm in m.named_modules():
        #         type_ = type(subm)
        #         if type_ in [nn.Sequential, nn.ModuleList]:
        #             continue
        #         is_leaf = is_leaf_module(subm)
        #         counter[type_].update(subm, name, is_leaf)

        def travel(m: nn.Module, name: str = ""):
            type_ = type(m)
            is_leaf = False
            if type_ not in [nn.Sequential, nn.ModuleList]:
                is_leaf = is_leaf_module(m, name)
                counter[type_].update(m, name, is_leaf)

            if not is_leaf:
                for sub_name, child in m.named_children():
                    travel(child, name + sub_name)

        travel(self.model)
        print(
            "\n----------------------------- Custom Summary -----------------------------"
        )

        table = []
        for type_, info in counter.items():
            # table.append({'type': type_, 'is_leaf': info.is_leaf, 'names': info.get_names(), 'duration': info.total_duration()})
            table.append(
                {
                    "type": type_,
                    "is_leaf": info.is_leaf,
                    "duration": info.total_duration(),
                }
            )
        print(tabulate(table, headers="keys"))

        # print leaf modules
        print(
            "\n----------------------------- Leaf Modules -----------------------------"
        )
        table = []
        leaf_modules = {k: v for k, v in counter.items() if v.is_leaf}
        for type_, info in sorted(
            leaf_modules.items(), key=lambda x: x[1].total_duration()
        ):
            table.append({"type": type_, "duration": info.total_duration()})
        print(tabulate(table, headers="keys"))

        self.model.apply(del_extra_repr)

        print(
            "------------------------------------------------------------------------------"
        )

        if output_file:
            sys.stdout = original_stdout
            f.close()

    def print_model_aggregated_profile(self, module_depth=-1, top_modules=1):
        """Prints the names of the top top_modules modules in terms of aggregated time, flops, and parameters at depth module_depth.

        Args:
            module_depth (int, optional): the depth of the modules to show. Defaults to -1 (the innermost modules).
            top_modules (int, optional): the number of top modules to show. Defaults to 1.
        """
        info = {}
        if not hasattr(self.model, "__flops__"):
            print(
                "no __flops__ attribute in the model, call this function after start_profile and before end_profile"
            )
            return

        def walk_module(module, curr_depth, info):
            if curr_depth not in info:
                info[curr_depth] = {}
            if module.__class__.__name__ not in info[curr_depth]:
                info[curr_depth][module.__class__.__name__] = [
                    0,
                    0,
                    0,
                ]  # macs, params, time
            info[curr_depth][module.__class__.__name__][0] += get_module_macs(
                module
            )
            info[curr_depth][module.__class__.__name__][1] += (
                module.__params__ + module.__expert_params__
            )
            info[curr_depth][module.__class__.__name__][
                2
            ] += get_module_duration(module)
            has_children = len(module._modules.items()) != 0
            if has_children:
                for child in module.children():
                    walk_module(child, curr_depth + 1, info)

        walk_module(self.model, 0, info)

        depth = module_depth
        if module_depth == -1:
            depth = len(info) - 1

        print(
            f"Top {top_modules} modules in terms of params, MACs or fwd latency at different model depths:"
        )

        for d in range(depth):
            num_items = min(top_modules, len(info[d]))

            sort_macs = {
                k: macs_to_string(v[0])
                for k, v in sorted(
                    info[d].items(), key=lambda item: item[1][0], reverse=True
                )[:num_items]
            }
            sort_params = {
                k: params_to_string(v[1])
                for k, v in sorted(
                    info[d].items(), key=lambda item: item[1][1], reverse=True
                )[:num_items]
            }
            sort_time = {
                k: duration_to_string(v[2])
                for k, v in sorted(
                    info[d].items(), key=lambda item: item[1][2], reverse=True
                )[:num_items]
            }

            print(f"depth {d}:")
            print(f"    params      - {sort_params}")
            print(f"    MACs        - {sort_macs}")
            print(f"    fwd latency - {sort_time}")


def get_model_profile(
    model,
    input_shape=None,
    args=[],
    kwargs={},
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    warm_up=1,
    as_string=True,
    output_file=None,
    ignore_modules=None,
    mode="forward",
    is_leaf_module=is_leaf_module,
):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Example:

    .. code-block:: python

        model = torchvision.models.alexnet()
        batch_size = 256
        flops, macs, params = get_model_profile(model=model, input_shape=(batch_size, 3, 224, 224)))

    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_shape (tuple): input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args (list): list of positional arguments to the model.
        kwargs (dict): dictionary of keyword arguments to the model.
        print_profile (bool, optional): whether to print the model profile. Defaults to True.
        detailed (bool, optional): whether to print the detailed model profile. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        warm_up (int, optional): the number of warm-up steps before measuring the latency of each module. Defaults to 1.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        output_file (str, optional): path to the output file. If None, the profiler prints to stdout.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    prof = YodaProfiler(model)
    model.eval()

    if input_shape is not None:
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert (
            len(input_shape) >= 1
        ), "input_shape must have at least one element"
        try:
            input = torch.ones(()).new_empty(
                (*input_shape,),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        except StopIteration:
            input = torch.ones(()).new_empty((*input_shape,))

        args = [input]
    assert (len(args) > 0) or (
        len(kwargs) > 0
    ), "args and/or kwargs must be specified if input_shape is None"

    logger.info("Flops profiler warming-up...")
    for _ in range(warm_up):
        if kwargs:
            if mode == "forward":
                _ = model(*args, **kwargs)
            if mode == "generate":
                _ = model.generate(*args, **kwargs)
        else:
            if mode == "forward":
                _ = model(*args)
            if mode == "generate":
                _ = model.generate(*args)
    prof.start_profile(ignore_list=ignore_modules)

    if kwargs:
        if mode == "forward":
            _ = model(*args, **kwargs)
        if mode == "generate":
            _ = model.generate(*args, **kwargs)
    else:
        if mode == "forward":
            _ = model(*args)
        if mode == "generate":
            _ = model.generate(*args)

    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(
            profile_step=warm_up,
            module_depth=module_depth,
            top_modules=top_modules,
            detailed=detailed,
            output_file=output_file,
            is_leaf_module=is_leaf_module,
        )

    prof.end_profile()
    if as_string:
        return (
            number_to_string(flops),
            macs_to_string(macs),
            params_to_string(params),
        )

    return flops, macs, params
