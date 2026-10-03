"""Shuffle-recovery benchmark (paper Table 1, Table 2, supplementary Tables 4-5).

Model B is model A with its hidden units randomly permuted; each method has to recover the
permutation. Reported per (model, method): strict L1 distance after applying the recovered hard
permutation, as ``log(1 + L1)``, precision (share of units mapped to their true position, in %)
and alignment time, as mean and standard deviation over seeds (MLP5 / VGG11 / ResNet18) or
scenes (NeRF, seed 42 per scene).

Example:
    python scripts/shuffle_recovery.py --models mlp5 nerf
    python scripts/shuffle_recovery.py --models vgg11 resnet18 --pretrained --out results/shuffle_pretrained
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppa import align, apply_permutation  # noqa: E402
from ppa.align import l1_distance  # noqa: E402
from ppa.baselines import ot4p, progressive_ot4p, progressive_weight_matching, weight_matching  # noqa: E402
from ppa.nerf import filter_nerf_params, load_state_dict, vanilla_nerf_permutation_spec  # noqa: E402
from ppa.ops import Params, perm_size  # noqa: E402
from ppa.spec import PermutationSpec  # noqa: E402

SEEDS = [42, 100, 200, 300, 400]
SCENES = ["lego", "chair", "ficus", "hotdog", "mic", "ship", "materials", "drums"]
METHODS = (
    ["wm", "prog-wm"] + [f"{p}ot4p-{t}" for p in ("", "prog-") for t in ("0.3", "0.5", "0.7")] + ["ours-no-ref", "ours"]
)
# Hyper-parameters of the runs behind the paper tables (they differ between the two code bases).
CONFIG = {
    "nerf": {"tol": 1e-6, "ot4p": dict(lr=0.05, max_iterations=100, patience=10, tol=1e-6),
             "prog-ot4p": dict(lr=0.1, max_iterations=100, patience=50, tol=1e-6)},
    "cnn": {"tol": 1e-8, "ot4p": dict(lr=0.1, max_iterations=100, patience=50, tol=1e-8, init_scale=1.0),
            "prog-ot4p": dict(lr=0.1, max_iterations=100, patience=15, tol=1e-8)},
}  # fmt: skip
TIMEOUT = 3600.0


def shuffle(spec: PermutationSpec, params: Params, seed: int) -> Tuple[Params, Dict[str, torch.Tensor]]:
    """Randomly permute every group (``torch.manual_seed(seed)``, groups in spec order)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gt = {pid: torch.randperm(perm_size(spec, params, pid)) for pid in spec.perm_to_axes}
    return apply_permutation(spec, params, gt), gt


def precision(found: Dict[str, torch.Tensor], gt: Dict[str, torch.Tensor]) -> float:
    """Share of units with ``found = argsort(gt)`` (the inverse of the shuffle), over all groups."""
    hits = sum(int((found[pid].cpu() == torch.argsort(gt[pid])).sum()) for pid in gt)
    return hits / sum(len(v) for v in gt.values())


def run_method(
    method: str,
    domain: str,
    spec: PermutationSpec,
    a: Params,
    b: Params,
    device: str,
    dtype: Optional[torch.dtype] = None,
    power_device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Recover the permutations of ``b`` with ``method`` (``dtype`` applies to the proposed method)."""
    cfg = CONFIG[domain]
    if method in ("ours", "ours-no-ref"):
        return align(spec, a, b, refine=method == "ours", tol=cfg["tol"], device=device, dtype=dtype).perms
    if method == "wm":
        return weight_matching(spec, a, b)
    if method == "prog-wm":
        return progressive_weight_matching(spec, a, b)
    kind, tau = method.rsplit("-", 1)
    fn: Callable = progressive_ot4p if kind == "prog-ot4p" else ot4p
    kw = dict(cfg[kind], power_device=power_device)
    return fn(spec, a, b, tau=float(tau), seed=0, device=device, timeout=TIMEOUT, **kw).perms


def cases(
    model: str, weights: Path, pretrained: bool = False, seeds: List[int] = SEEDS
) -> List[Tuple[str, PermutationSpec, Params, int]]:
    """``(case name, spec, model A parameters, shuffle seed)`` for one model family.

    With ``pretrained`` model A is the same ImageNet model for every seed; only the shuffle changes.
    """
    if model == "nerf":
        spec = vanilla_nerf_permutation_spec()
        return [(s, spec, filter_nerf_params(load_state_dict(weights / s / "model_a.pth")), 42) for s in SCENES]
    from ppa import zoo

    out = []
    for seed in seeds:
        torch.manual_seed(seed)
        module, spec = zoo.build(model, pretrained=pretrained)
        out.append((f"seed{seed}", spec, zoo.state_params(module), seed))
    return out


def main() -> None:
    """Parse arguments, run the benchmark and write ``shuffle_recovery.json``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models", nargs="+", default=["mlp5", "resnet18", "nerf"], choices=["mlp5", "vgg11", "resnet18", "nerf"]
    )
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parents[1] / "weights")
    parser.add_argument("--out", type=Path, default=Path("results/shuffle"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true", help="ImageNet weights for VGG11 / ResNet18")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--skip", nargs="*", default=[], help="runs to skip, as model:case:method")
    parser.add_argument(
        "--ot4p-power-device",
        default=None,
        help="device of the OT4P eigendecomposition power (default: as the official code, the input device; "
        "use cpu on GPUs that cannot JIT-compile complex kernels, e.g. RTX 50xx with CUDA 11.8 wheels)",
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float32", help="precision of the proposed method"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "shuffle_recovery.json"
    records = json.loads(path.read_text()) if path.exists() else []
    done = {(r["model"], r["case"], r["method"]) for r in records}
    for base_model in args.models:
        domain = "nerf" if base_model == "nerf" else "cnn"
        pretrained = args.pretrained and base_model in ("vgg11", "resnet18")
        model = f"{base_model}-pretrained" if pretrained else base_model
        for case, spec, a, seed in cases(base_model, args.weights, pretrained, args.seeds):
            b, gt = shuffle(spec, a, seed)
            for method in args.methods:
                if (model, case, method) in done or f"{model}:{case}:{method}" in args.skip:
                    continue
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                perms = run_method(
                    method, domain, spec, a, b, args.device, getattr(torch, args.dtype), args.ot4p_power_device
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                aligned = apply_permutation(spec, b, {k: v.cpu() for k, v in perms.items()})
                rec = {
                    "model": model,
                    "case": case,
                    "method": method,
                    "time_sec": elapsed,
                    "l1": l1_distance(a, aligned),
                    "precision": precision(perms, gt),
                    "timed_out": elapsed >= TIMEOUT,
                }
                records.append(rec)
                path.write_text(json.dumps(records, indent=1))
                print(f"{model:8s} {case:9s} {method:14s} log1p(L1)={math.log1p(rec['l1']):7.3f} "
                      f"prec={100 * rec['precision']:6.2f}% time={elapsed:8.2f}s", flush=True)  # fmt: skip

    print("\nmean ± std")
    for model in dict.fromkeys(r["model"] for r in records):  # models in the order they were run
        for method in args.methods:
            rs = [r for r in records if r["model"] == model and r["method"] == method]
            if not rs:
                continue
            l1 = [math.log1p(r["l1"]) for r in rs]
            pr = [100 * r["precision"] for r in rs]
            tm = [r["time_sec"] for r in rs]
            sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0  # noqa: E731
            print(f"{model:8s} {method:14s} L1 {np.mean(l1):7.3f}±{sd(l1):.3f}  prec {np.mean(pr):6.2f}±{sd(pr):.3f}  "
                  f"time {np.mean(tm):8.2f}±{sd(tm):.2f}  (n={len(rs)})")  # fmt: skip


if __name__ == "__main__":
    main()
