import os
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torchview import ComputationGraph, draw_graph

from yoda.utils.path_utils import create_parent_dir


def visualize(
    module: nn.Module,
    input_data: Dict[str, Union[torch.Tensor, Any]],
    depth: int = 0,
    hide_inner_tensors: bool = True,
    expand_nested: bool = True,
    save_path: Optional[str] = None,
) -> ComputationGraph:
    """visualize the given module"""
    if save_path:
        save_graph = True
        create_parent_dir(save_path)
    else:
        save_graph = False

    graph = draw_graph(
        module,
        input_data,
        depth=depth,
        hide_inner_tensors=hide_inner_tensors,
        expand_nested=expand_nested,
        save_graph=save_graph,
        filename=save_path,
    )
    # clean meta for graphviz
    if save_graph:
        os.remove(save_path)

    return graph
