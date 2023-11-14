import pytest
import torch
import torchvision
from torch import nn

from yoda.surgeon.fusion_pass import fuse_conv_bn_pass, fuse_linear_bn_pass


def test_conv_bn_fusion():
    x = torch.rand(1, 3, 224, 224)
    net = torchvision.models.resnet18(pretrained=False).eval()
    with torch.no_grad():
        expected = net(x)

    fused_net = fuse_conv_bn_pass(net)
    with torch.no_grad():
        out = fused_net(x)
    torch.testing.assert_close(out, expected)
    # bn has been eliminated
    has_bn = any(
        isinstance(x[1], nn.BatchNorm2d) for x in fused_net.named_modules()
    )
    assert not has_bn


def test_linear_bn_fusion():
    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(20, 40),
            )

        def forward(self, x):
            return self.net(x)

    net = Module().eval()
    x = torch.rand(1, 10)

    with torch.no_grad():
        expected = net(x)

    fused_net = fuse_linear_bn_pass(net)
    with torch.no_grad():
        out = fused_net(x)

    torch.testing.assert_close(out, expected)

    has_bn = any(
        isinstance(x[1], nn.BatchNorm1d) for x in fused_net.named_modules()
    )
    assert not has_bn
