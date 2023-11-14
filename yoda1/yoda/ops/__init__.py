import inspect
from contextlib import contextmanager
from importlib import import_module
from types import ModuleType
from typing import Dict, List, Tuple, Type

from loguru import logger
from torch import nn

from yoda.utils.check_utils import check_types

_TorchModuleType = Type[nn.Module]


class MonkeypatchManager:
    def __init__(self):
        # it is a mapping from class type to a tuple of (its module, its name)
        # for example, for torch.nn.Linear
        # {torch.nn.Linear: (torch.nn, 'Linear')}
        self.container: Dict[
            _TorchModuleType, Tuple[str, List[ModuleType]]
        ] = {}
        # old type to new type map
        self.type_map: Dict[_TorchModuleType, _TorchModuleType] = {}

    def get_all_modified_types(self) -> List[_TorchModuleType]:
        """get all types which are modified"""
        return list(self.container.keys())

    def add(self, type_: _TorchModuleType, new_type: _TorchModuleType):
        module = inspect.getmodule(type_)
        name = type_.__name__
        type_: _TorchModuleType = getattr(module, name)
        assert (
            type_ not in self.container
        ), f"{module.__name__}.{name} has been registered"
        self.type_map[type_] = new_type

        # module which will be modified
        module_hierachy = module.__name__.split(".")
        modules = []
        for i in range(len(module_hierachy), 1, -1):
            cur_module = import_module(".".join(module_hierachy[:i]))
            if hasattr(cur_module, name):
                modules.append(cur_module)
                logger.info(
                    f"Replace {cur_module.__name__}.{name} with new type {new_type}"
                )
                setattr(cur_module, name, new_type)
        self.container[type_] = (name, modules)

    def remove(self, type_: _TorchModuleType):
        for key in self.container:
            if key is type_:
                name, modules = self.container[key]
                for module in modules:
                    logger.info(f"Restore {module.__name__}.{name}")
                    setattr(module, name, key)

    def remove_all(self):
        for type_ in self.container:
            self.remove(type_)

    def check(self, module: nn.Module):
        check_types(module, nn.Module)

        for name, submodule in module.named_modules():
            for type_ in self.container:
                if isinstance(submodule, type_) and type(submodule) is type_:
                    raise RuntimeError(
                        f"Find module: {name} is not replaced, type: {type_}"
                    )


@contextmanager
def build_model_with_yoda_ops(
    enable_flash_atten_in_cross_atten: bool,
    enable_flash_atten_in_temporal_atten: bool,
):

    manager = MonkeypatchManager()

    if enable_flash_atten_in_cross_atten:
        from diffusers.models.attention import CrossAttention

        from .cross_attn_with_flash_atten import CrossAttentionFlashAttn

        manager.add(CrossAttention, CrossAttentionFlashAttn)

    if enable_flash_atten_in_temporal_atten:
        from libs.models.motion_module import VersatileAttention

        from .temporal_atten_with_flash_atten import VersatileAttentionFlashAttn

        manager.add(VersatileAttention, VersatileAttentionFlashAttn)

    yield manager

    manager.remove_all()
