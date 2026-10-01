"""Rendering of vanilla NeRF weights on NeRF Synthetic test views (requires nerfstudio).

:class:`VanillaNeRF` re-implements the forward pass of nerfstudio's ``vanilla-nerf`` model with
plain ``nn.Linear`` layers so that arbitrary (e.g. interpolated) state dicts can be loaded.
Samplers, renderers and ray generation are nerfstudio's (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig

from ppa.ops import Params

NUM_TEST_VIEWS = 10
VIEW_SEED = 42


@dataclass
class VanillaNeRFConfig(ModelConfig):
    """Configuration matching ``ns-train vanilla-nerf`` (64 coarse + 128 importance samples)."""

    _target: Type = field(default_factory=lambda: VanillaNeRF)
    num_coarse_samples: int = 64
    num_importance_samples: int = 128
    background_color: Literal["random", "last_sample", "black", "white"] = "white"


def _mlp_base() -> nn.ModuleList:
    return nn.ModuleList(
        [nn.Linear(63, 256)]
        + [nn.Linear(256, 256) for _ in range(3)]
        + [nn.Linear(256 + 63, 256)]
        + [nn.Linear(256, 256) for _ in range(3)]
    )


class _Field(nn.Module):
    """One NeRF MLP (coarse or fine) with nerfstudio's parameter names."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp_base = nn.Module()
        self.mlp_base.layers = _mlp_base()
        self.field_output_density = nn.Module()
        self.field_output_density.net = nn.Linear(256, 1)
        self.mlp_head = nn.Module()
        self.mlp_head.layers = nn.ModuleList([nn.Linear(256 + 27, 128), nn.Linear(128, 128)])
        self.field_heads = nn.ModuleList([nn.Module()])
        self.field_heads[0].net = nn.Linear(128, 3)
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rgb, density)`` at sample positions."""
        enc = self.position_encoding(positions)
        x = enc
        for i, layer in enumerate(self.mlp_base.layers):
            if i == 4:
                x = torch.cat([enc, x], dim=-1)
            x = F.relu(layer(x))
        density = F.softplus(self.field_output_density.net(x))
        x = torch.cat([self.direction_encoding(directions), x], dim=-1)
        for layer in self.mlp_head.layers:
            x = F.relu(layer(x))
        return torch.sigmoid(self.field_heads[0].net(x)), density


class VanillaNeRF(Model):
    """Hierarchical vanilla NeRF whose ``state_dict`` keys match nerfstudio's ``field_*`` keys."""

    config: VanillaNeRFConfig

    def populate_modules(self) -> None:
        """Create fields, samplers and renderer."""
        super().populate_modules()
        self.field_coarse = _Field()
        self.field_fine = _Field()
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Unused (inference only)."""
        return {"fields": list(self.parameters())}

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Render coarse and fine RGB for a ray bundle."""
        samples = self.sampler_uniform(ray_bundle)
        rgb, density = self.field_coarse(samples.frustums.get_positions(), samples.frustums.directions)
        weights_coarse = samples.get_weights(density)
        rgb_coarse = self.renderer_rgb(rgb=rgb, weights=weights_coarse)
        samples_pdf = self.sampler_pdf(ray_bundle, samples, weights_coarse)
        rgb, density = self.field_fine(samples_pdf.frustums.get_positions(), samples_pdf.frustums.directions)
        rgb_fine = self.renderer_rgb(rgb=rgb, weights=samples_pdf.get_weights(density))
        return {"rgb_coarse": rgb_coarse, "rgb_fine": rgb_fine}


@dataclass
class TestViews:
    """Test split of a NeRF Synthetic scene."""

    dataset: InputDataset
    scene_box: SceneBox
    indices: List[int]


def load_test_views(data_dir: Union[str, Path], num_views: int = NUM_TEST_VIEWS, seed: int = VIEW_SEED) -> TestViews:
    """Load the test split and pick the evaluation views used in the paper.

    The views are ``sorted(RandomState(seed).choice(n_test, num_views, replace=False))``.

    Args:
        data_dir: Scene directory of NeRF Synthetic (containing ``transforms_test.json``).
        num_views: Number of views.
        seed: Selection seed.

    Returns:
        The test views.
    """
    parser = BlenderDataParserConfig(data=Path(data_dir)).setup()
    outputs = parser.get_dataparser_outputs(split="test")
    dataset = InputDataset(outputs)
    n = len(dataset)
    indices = (
        sorted(np.random.RandomState(seed).choice(n, size=num_views, replace=False).tolist())
        if num_views < n
        else list(range(n))
    )
    return TestViews(dataset=dataset, scene_box=outputs.scene_box, indices=indices)


def build_model(params: Params, scene_box: SceneBox, device: str) -> VanillaNeRF:
    """Instantiate :class:`VanillaNeRF` with the given (nerfstudio-keyed) weights.

    Args:
        params: State dict; only ``field_coarse.*`` / ``field_fine.*`` entries are used.
        scene_box: Scene box of the dataset.
        device: Torch device.

    Returns:
        The model in eval mode.

    Raises:
        KeyError: If a field parameter is missing.
    """
    model = VanillaNeRF(config=VanillaNeRFConfig(), scene_box=scene_box, num_train_data=100)
    own = model.state_dict()
    needed = [k for k in own if k.startswith(("field_coarse.", "field_fine.")) and "encoding" not in k]
    missing = [k for k in needed if k not in params]
    if missing:
        raise KeyError(f"missing NeRF parameters: {missing[:3]}...")
    model.load_state_dict({k: params[k] for k in needed}, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def render_view(model: VanillaNeRF, views: TestViews, idx: int, device: str, chunk: int = 4096) -> torch.Tensor:
    """Render test view ``idx`` (fine network) as an ``(H, W, 3)`` tensor in ``[0, 1]``."""
    c = views.dataset.cameras
    cam = Cameras(
        fx=c.fx[idx : idx + 1],
        fy=c.fy[idx : idx + 1],
        cx=c.cx[idx : idx + 1],
        cy=c.cy[idx : idx + 1],
        height=c.height[idx : idx + 1],
        width=c.width[idx : idx + 1],
        camera_to_worlds=c.camera_to_worlds[idx : idx + 1],
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)
    h, w = int(cam.height.item()), int(cam.width.item())
    bundle = cam.generate_rays(camera_indices=0, keep_shape=True, aabb_box=views.scene_box)
    out = [
        model.get_outputs(bundle.get_row_major_sliced_ray_bundle(i, min(i + chunk, len(bundle))))["rgb_fine"]
        for i in range(0, len(bundle), chunk)
    ]
    return torch.cat(out, dim=0).view(h, w, 3)


def ground_truth(views: TestViews, idx: int) -> torch.Tensor:
    """Test image ``idx`` composited on white, ``(H, W, 3)`` in ``[0, 1]``."""
    return views.dataset[idx]["image"][..., :3]


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a ``[0, 1]`` float image to ``uint8`` exactly as the paper's PNGs were written."""
    return (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR in dB for images in ``[0, 1]``."""
    mse = F.mse_loss(pred, gt).item()
    return 100.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def rmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Root mean squared error for images in ``[0, 1]``."""
    return float(np.sqrt(F.mse_loss(pred, gt).item()))


def lambdas(step: float = 0.1) -> Sequence[float]:
    """Interpolation coefficients ``0, step, ..., 1`` rounded to one decimal as in the paper."""
    n = int(round(1 / step))
    return [round(i * step, 1) for i in range(n + 1)]
