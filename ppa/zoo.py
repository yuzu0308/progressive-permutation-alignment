"""Models and permutation specs of the shuffle-recovery experiments (paper Table 1).

The specs follow the ones used for the paper. Two simplifications there are kept as is, since the
experiment measures recovery of a known shuffle in weight space rather than function preservation:
the last VGG11 conv permutation is not propagated into the flattened classifier input, and the
ResNet18 ``layer1`` blocks use separate input/output permutations although their identity skip
ties them together. A shuffled model B is therefore not always functionally identical to A.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ppa.spec import PermutationSpec, permutation_spec_from_axes_to_perm


class MLP5(nn.Module):
    """5-layer MLP (784-256-256-256-256-10) with ReLU."""

    def __init__(self, in_features: int = 784, hidden: int = 256, out_features: int = 10) -> None:
        """Create the layers."""
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in (self.fc1, self.fc2, self.fc3, self.fc4):
            x = torch.relu(layer(x))
        return self.fc5(x)


def mlp5_spec() -> PermutationSpec:
    """Permutation spec of :class:`MLP5` (groups ``P_fc1`` ... ``P_fc4``)."""
    axes: Dict[str, List[Optional[str]]] = {"fc1.weight": ["P_fc1", None], "fc1.bias": ["P_fc1"]}
    for i in range(2, 5):
        axes[f"fc{i}.weight"] = [f"P_fc{i}", f"P_fc{i - 1}"]
        axes[f"fc{i}.bias"] = [f"P_fc{i}"]
    axes["fc5.weight"] = [None, "P_fc4"]
    axes["fc5.bias"] = [None]
    return permutation_spec_from_axes_to_perm(axes)


def vgg11_spec() -> PermutationSpec:
    """Permutation spec of torchvision's ``vgg11`` (8 conv groups + 2 classifier groups)."""
    conv = [0, 3, 6, 8, 11, 13, 16, 18]
    axes: Dict[str, List[Optional[str]]] = {
        "features.0.weight": ["P_conv0", None, None, None],
        "features.0.bias": ["P_conv0"],
    }
    for i, idx in enumerate(conv[1:], start=1):
        axes[f"features.{idx}.weight"] = [f"P_conv{i}", f"P_conv{i - 1}", None, None]
        axes[f"features.{idx}.bias"] = [f"P_conv{i}"]
    axes["classifier.0.weight"] = ["P_fc0", None]
    axes["classifier.0.bias"] = ["P_fc0"]
    axes["classifier.3.weight"] = ["P_fc1", "P_fc0"]
    axes["classifier.3.bias"] = ["P_fc1"]
    axes["classifier.6.weight"] = [None, "P_fc1"]
    axes["classifier.6.bias"] = [None]
    return permutation_spec_from_axes_to_perm(axes)


def _bn(axes: Dict[str, List[Optional[str]]], prefix: str, perm: str) -> None:
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        axes[f"{prefix}.{suffix}"] = [perm]


def resnet18_spec() -> PermutationSpec:
    """Permutation spec of torchvision's ``resnet18`` (BatchNorm statistics included)."""
    axes: Dict[str, List[Optional[str]]] = {"conv1.weight": ["P_stem", None, None, None]}
    _bn(axes, "bn1", "P_stem")
    layers: List[Tuple[str, str, str, bool]] = [
        ("layer1", "P_stem", "P_layer1", False),
        ("layer2", "P_layer1", "P_layer2", True),
        ("layer3", "P_layer2", "P_layer3", True),
        ("layer4", "P_layer3", "P_layer4", True),
    ]
    for name, in_perm, out_perm, downsample in layers:
        for b in range(2):
            pre, block_in = f"{name}.{b}", (in_perm if b == 0 else out_perm)
            inner = f"P_{name}_{b}_internal"
            axes[f"{pre}.conv1.weight"] = [inner, block_in, None, None]
            _bn(axes, f"{pre}.bn1", inner)
            axes[f"{pre}.conv2.weight"] = [out_perm, inner, None, None]
            _bn(axes, f"{pre}.bn2", out_perm)
            if b == 0 and downsample:
                axes[f"{pre}.downsample.0.weight"] = [out_perm, block_in, None, None]
                _bn(axes, f"{pre}.downsample.1", out_perm)
    axes["fc.weight"] = [None, "P_layer4"]
    axes["fc.bias"] = [None]
    return permutation_spec_from_axes_to_perm(axes)


def build(model: str, pretrained: bool = False) -> Tuple[nn.Module, PermutationSpec]:
    """Model and its spec.

    The paper runs used randomly initialised models (seed the global torch RNG beforehand).
    ``pretrained=True`` loads torchvision's ImageNet weights (``IMAGENET1K_V1``) instead.

    Args:
        model: ``mlp5``, ``vgg11`` or ``resnet18``.
        pretrained: Use ImageNet weights (VGG11 / ResNet18 only).

    Returns:
        ``(module, spec)``.

    Raises:
        ValueError: For an unknown model name, or ``pretrained`` with ``mlp5``.
    """
    if pretrained and model == "mlp5":
        raise ValueError("no pretrained weights for mlp5")
    factories: Dict[str, Callable[[], Tuple[nn.Module, PermutationSpec]]] = {"mlp5": lambda: (MLP5(), mlp5_spec())}
    if model in ("vgg11", "resnet18"):
        import torchvision.models as tvm

        vgg_w = tvm.VGG11_Weights.IMAGENET1K_V1 if pretrained else None
        res_w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        factories["vgg11"] = lambda: (tvm.vgg11(weights=vgg_w), vgg11_spec())
        factories["resnet18"] = lambda: (tvm.resnet18(weights=res_w), resnet18_spec())
    if model not in factories:
        raise ValueError(f"unknown model {model}")
    return factories[model]()


def state_params(module: nn.Module) -> Dict[str, torch.Tensor]:
    """State dict without ``num_batches_tracked`` counters."""
    return {k: v.clone() for k, v in module.state_dict().items() if "num_batches_tracked" not in k}
