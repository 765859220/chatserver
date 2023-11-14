import os

import torch

from yoda.ops import MonkeypatchManager
from yoda.utils.path_utils import create_tmp_dir


def test_replace_linear_with_relu():
    from torch.nn import Linear, ReLU

    module = Linear(10, 20)
    tmp_dir = create_tmp_dir()
    filename = os.path.join(tmp_dir, "test_linear.pth")
    torch.save(module, filename)

    old_Linear = Linear
    m = MonkeypatchManager()

    # replace linear with relu
    m.add(Linear, ReLU)
    from torch.nn import Linear

    assert Linear is ReLU
    assert Linear.__name__ == "ReLU"

    # also works for loading checkpoint
    module_back = torch.load(filename)
    assert isinstance(module_back, ReLU)

    from torch.nn.modules.linear import Linear

    assert Linear is ReLU
    assert Linear.__name__ == "ReLU"
    assert not isinstance(module, Linear)

    # restore
    m.remove(old_Linear)
    from torch.nn import Linear

    assert Linear is old_Linear
    assert Linear.__name__ == "Linear"
    assert isinstance(module, Linear)
    module_back = torch.load(filename)
    assert isinstance(module_back, Linear)


def test_replace_relu_with_sigmoid():
    from torch.nn import ReLU, Sigmoid

    m = MonkeypatchManager()
    module = ReLU()
    tmp_dir = create_tmp_dir()
    filename = os.path.join(tmp_dir, "test_linear.pth")
    torch.save(module, filename)

    m.add(ReLU, Sigmoid)
    module_back = torch.load(filename)

    x = torch.tensor(0.0)
    out = module(x)
    out_modified = module_back(x)
    assert out.item() == 0
    assert out_modified.item() == 0.5
