import pytest
import torch

from yoda.utils import torch_utils


def test_tensor_device():
    x = torch.tensor(1.0)
    assert torch_utils.get_device(x) == torch.device("cpu")

    if torch.cuda.is_available():
        x = x.to("cuda:0")
        assert torch_utils.get_device(x) == torch.device("cuda:0")


def test_module_device():
    m = torch.nn.ReLU()
    assert torch_utils.get_device(m) == torch.device("cpu")

    if torch.cuda.is_available():
        m = m.to("cuda:0")
        # module which has no parameters or buffers will be judged in cpu
        assert torch_utils.get_device(m) == torch.device("cpu")

    m = torch.nn.Linear(10, 20)
    assert torch_utils.get_device(m) == torch.device("cpu")

    if torch.cuda.is_available():
        m = m.to("cuda:0")
        assert torch_utils.get_device(m) == torch.device("cuda:0")


def test_reduce_tensor():
    x = torch.rand(3, 4)
    xmax = torch_utils.reduce_max(x)
    torch.testing.assert_close(xmax, x.max())

    xmin = torch_utils.reduce_min(x)
    torch.testing.assert_close(xmin, x.min())


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_reduce_axis(axis):
    def reduce(x, op):
        return op(
            torch.reshape(
                torch.transpose(x, dim0=axis, dim1=-1), (-1, x.shape[axis])
            ),
            dim=0,
            keepdim=False,
        )[0]

    x = torch.rand(3, 4, 2, 2)

    # max
    xmax = torch_utils.reduce_max(x, excluded_axis=axis)
    expected = reduce(x, torch.max)
    torch.testing.assert_close(xmax, expected=expected)

    xmax = torch_utils.reduce_max(x, excluded_axis=axis, keepdim=True)
    assert list(xmax.shape) == [
        {axis: x.shape[axis]}.get(i, 1) for i in range(x.ndim)
    ]
    torch.testing.assert_close(xmax.squeeze(), expected.squeeze())

    # min
    xmin = torch_utils.reduce_min(x, excluded_axis=axis)
    expected = reduce(x, torch.min)
    torch.testing.assert_close(xmin, expected=expected)

    xmin = torch_utils.reduce_min(x, excluded_axis=axis, keepdim=True)
    assert list(xmin.shape) == [
        {axis: x.shape[axis]}.get(i, 1) for i in range(x.ndim)
    ]
    torch.testing.assert_close(xmin.squeeze(), expected.squeeze())
