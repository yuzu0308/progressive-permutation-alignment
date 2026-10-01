"""Data-dependent baselines of the supplementary material (Table 1). Require nerfstudio.

* :func:`activation_matching` ("AM"): Hungarian matching of ``E[a_A a_B^T]`` of every hidden
  layer's pre-activation, accumulated over ``num_batches`` training ray batches.
* :func:`ste_alignment` ("STE"): straight-through estimator (Ainsworth et al., 2023). A copy of
  model A is trained so that the midpoint ``0.5 (A + P(B))`` renders the training images, where
  ``P`` is recomputed every ``update_perm_every`` steps by Progressive WM between the copy and B.

Both use nerfstudio's training data manager to draw rays, so their results depend on the global
torch RNG state; call ``torch.manual_seed`` beforehand for repeatable runs.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ppa.baselines.weight_matching import progressive_weight_matching
from ppa.nerf import filter_nerf_params
from ppa.ops import Params, apply_permutation
from ppa.spec import PermutationSpec
from ppa.spectral import hungarian


def make_pipeline(data_dir: Any, device: str = "cuda") -> Any:
    """nerfstudio ``vanilla-nerf`` pipeline on a NeRF Synthetic scene (no checkpoint loaded).

    Only its data manager (training rays) and its model (used as an architecture template whose
    weights are overwritten) are needed by the baselines.

    Args:
        data_dir: Scene directory.
        device: Torch device.

    Returns:
        The pipeline in eval mode.
    """
    from pathlib import Path

    from nerfstudio.configs.method_configs import method_configs

    config = copy.deepcopy(method_configs["vanilla-nerf"])
    config.pipeline.datamanager.data = Path(data_dir)
    config.pipeline.datamanager.dataparser.data = Path(data_dir)
    pipeline = config.pipeline.setup(device=torch.device(device), test_mode="test")
    pipeline.eval()
    return pipeline


def _next_train(datamanager: Any, step: int):
    out = datamanager.next_train(step)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    raise ValueError("unsupported datamanager.next_train output")


def _collide(model: Any, ray_bundle: Any) -> Any:
    if getattr(ray_bundle, "nears", None) is None and getattr(model, "collider", None) is not None:
        ray_bundle = model.collider(ray_bundle)
    return ray_bundle


def _hidden_modules(spec: PermutationSpec) -> Dict[str, str]:
    """Permutation id -> name of the ``nn.Linear`` whose output neurons it permutes."""
    out = {}
    for pid, axes in spec.perm_to_axes.items():
        for name, axis in axes:
            if axis == 0 and name.endswith(".weight"):
                out[pid] = name[: -len(".weight")]
                break
    return out


def _load_model(template: torch.nn.Module, params: Params, device: str) -> torch.nn.Module:
    model = copy.deepcopy(template)
    model.load_state_dict(params, strict=False)
    return model.to(device).eval()


def activation_matching(
    spec: PermutationSpec,
    pipeline: Any,
    params_a: Params,
    params_b: Params,
    num_batches: int = 20,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """AM: match hidden units by their activation inner products on training rays.

    Args:
        spec: Permutation spec (vanilla NeRF).
        pipeline: nerfstudio pipeline from ``eval_setup`` (provides the model template and rays).
        params_a: Full state dict of model A.
        params_b: Full state dict of model B.
        num_batches: Number of training ray batches.
        device: Torch device.

    Returns:
        Permutations (index vectors on ``device``).
    """
    model_a = _load_model(pipeline.model, params_a, device)
    model_b = _load_model(pipeline.model, params_b, device)
    modules = _hidden_modules(spec)
    stores: List[Dict[str, torch.Tensor]] = [{}, {}]
    handles = []
    for model, store in ((model_a, stores[0]), (model_b, stores[1])):
        named = dict(model.named_modules())
        for pid, mod in modules.items():

            def hook(_m: Any, _i: Any, out: Any, pid: str = pid, store: Dict[str, torch.Tensor] = store) -> None:
                store[pid] = (out[0] if isinstance(out, (tuple, list)) else out).detach()

            handles.append(named[mod].register_forward_hook(hook))

    stats: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    try:
        for step in range(num_batches):
            ray_bundle, _ = _next_train(pipeline.datamanager, step)
            ray_bundle = ray_bundle.to(device)
            for model, store in ((model_a, stores[0]), (model_b, stores[1])):
                store.clear()
                with torch.no_grad():
                    model.get_outputs(_collide(model, ray_bundle))
            for pid in modules:
                act_a = stores[0][pid].reshape(-1, stores[0][pid].shape[-1]).double()
                act_b = stores[1][pid].reshape(-1, stores[1][pid].shape[-1]).double()
                if pid not in stats:
                    stats[pid] = torch.zeros(act_a.shape[-1], act_b.shape[-1], device=act_a.device, dtype=torch.float64)
                    counts[pid] = 0
                stats[pid] += act_a.T @ act_b
                counts[pid] += act_a.shape[0]
    finally:
        for h in handles:
            h.remove()
    return {pid: hungarian(stats[pid] / counts[pid]) for pid in stats}


def ste_alignment(
    spec: PermutationSpec,
    pipeline: Any,
    params_a: Params,
    params_b: Params,
    num_steps: int = 200,
    update_perm_every: int = 10,
    lr: float = 1e-3,
    lam: float = 0.5,
    device: str = "cuda",
    log: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    """STE: learn the permutation through a trainable proxy of model A.

    Args:
        spec: Permutation spec (vanilla NeRF).
        pipeline: nerfstudio pipeline from ``eval_setup``.
        params_a: Full state dict of model A.
        params_b: Full state dict of model B.
        num_steps: SGD steps.
        update_perm_every: Recompute the permutation every this many steps.
        lr: SGD learning rate (momentum 0.9).
        lam: Interpolation weight of the rendered model.
        device: Torch device.
        log: Optional logging callback.

    Returns:
        The last permutation (index vectors on CPU).
    """
    try:
        from torch.func import functional_call
    except ImportError:  # torch < 2.0
        from torch.nn.utils.stateless import functional_call  # type: ignore[no-redef]

    model_a = _load_model(pipeline.model, params_a, device)
    model_b = _load_model(pipeline.model, params_b, device)
    for p in list(model_a.parameters()) + list(model_b.parameters()):
        p.requires_grad = False
    full_a = model_a.state_dict()
    proxy = {k: v.clone().detach().to(device).requires_grad_(True) for k, v in filter_nerf_params(full_a).items()}
    optimizer = torch.optim.SGD(proxy.values(), lr=lr, momentum=0.9)
    b_fixed = filter_nerf_params(model_b.state_dict())

    perms: Optional[Dict[str, torch.Tensor]] = None
    for step in range(num_steps):
        ray_bundle, batch = _next_train(pipeline.datamanager, step)
        ray_bundle = _collide(model_a, ray_bundle.to(device))
        gt = batch["image"].to(device)
        if perms is None or step % update_perm_every == 0:
            perms = progressive_weight_matching(spec, {k: v.detach() for k, v in proxy.items()}, b_fixed)
        permuted_b = apply_permutation(spec, b_fixed, perms)
        merged = dict(full_a)
        for k, va in full_a.items():
            if k in permuted_b and k in proxy:
                ste_b = permuted_b[k].detach() + (proxy[k] - proxy[k].detach())
                merged[k] = (1.0 - lam) * va + lam * ste_b
        state = {**dict(model_a.named_parameters()), **dict(model_a.named_buffers()), **merged}
        try:
            outputs = functional_call(model_a, state, (ray_bundle,), tie_weights=False)
        except TypeError:  # torch versions without ``tie_weights``
            outputs = functional_call(model_a, state, (ray_bundle,))
        loss = F.mse_loss(outputs["rgb_fine"] if "rgb_fine" in outputs else outputs["rgb"], gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if log is not None and (step + 1) % max(1, num_steps // 10) == 0:
            log(f"[ste] step={step + 1}/{num_steps} loss={loss.item():.6f}")
    assert perms is not None
    return {k: v.cpu() for k, v in perms.items()}
