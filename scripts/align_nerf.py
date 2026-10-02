"""Align model B of a NeRF pair to model A with any method of the paper.

Example:
    python scripts/align_nerf.py --model-a weights/lego/model_a.pth --model-b weights/lego/model_b.pth \
        --method ours --out results/lego/ours

Methods:
    ours, ours-no-ref                     proposed (Table 3)
    ours-no-svd, ours-no-svd-no-ref       "w/o SVD" ablation (supp. Table 2)
    ours-tau2[-no-ref], ours-tau5[-no-ref]  tau schedules {1,0} / {1,.75,.5,.25,0} (supp. Table 3)
    naive                                 no alignment
    wm, prog-wm                           Weight Matching / Progressive WM
    ot4p-0.3|0.5|0.7, prog-ot4p-0.3|0.5|0.7  OT4P / Progressive OT4P at fixed tau
    am, ste                               data-dependent baselines (supp. Table 1, need --data)

Writes ``<out>/model_b_aligned.pth`` (full state dict) and ``<out>/alignment.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppa import align, apply_permutation  # noqa: E402
from ppa.align import l1_distance, squared_distance  # noqa: E402
from ppa.nerf import filter_nerf_params, load_state_dict, vanilla_nerf_permutation_spec  # noqa: E402
from ppa.ops import Params, apply_matrices  # noqa: E402

# name -> (refine, use_svd, tau schedule)
OURS: Dict[str, Tuple[bool, bool, Tuple[float, ...]]] = {
    "ours": (True, True, (1.0, 0.5, 0.0)),
    "ours-no-ref": (False, True, (1.0, 0.5, 0.0)),
    "ours-no-svd": (True, False, (1.0, 0.5, 0.0)),
    "ours-no-svd-no-ref": (False, False, (1.0, 0.5, 0.0)),
    "ours-tau2": (True, True, (1.0, 0.0)),
    "ours-tau2-no-ref": (False, True, (1.0, 0.0)),
    "ours-tau5": (True, True, (1.0, 0.75, 0.5, 0.25, 0.0)),
    "ours-tau5-no-ref": (False, True, (1.0, 0.75, 0.5, 0.25, 0.0)),
}
OT4P_METHODS = [f"{p}ot4p-{t}" for p in ("", "prog-") for t in ("0.3", "0.5", "0.7")]
METHODS = sorted(list(OURS) + ["naive", "wm", "prog-wm", "am", "ste"] + OT4P_METHODS)


def run(method: str, full_a: Params, full_b: Params, args: argparse.Namespace) -> Tuple[Params, Dict[str, object]]:
    """Align ``full_b`` to ``full_a`` with ``method``; return the aligned state dict and extra info."""
    spec = vanilla_nerf_permutation_spec()
    a, b = filter_nerf_params(full_a), filter_nerf_params(full_b)
    log = print if args.verbose else None
    if method in OURS:
        refine, use_svd, schedule = OURS[method]
        r = align(spec, a, b, tau_schedule=schedule, refine=refine, use_svd=use_svd, device=args.device, log=log)
        info = {"tau_schedule": list(schedule), "refine": refine, "use_svd": use_svd}
        return apply_permutation(spec, full_b, r.perms), info
    if method == "naive":
        return dict(full_b), {}
    if method in ("wm", "prog-wm"):
        from ppa.baselines import progressive_weight_matching, weight_matching

        fn = weight_matching if method == "wm" else progressive_weight_matching
        return apply_permutation(spec, full_b, fn(spec, a, b, log=log)), {}
    if method in OT4P_METHODS:
        from ppa.baselines import ot4p, progressive_ot4p

        prog, tau = method.startswith("prog-"), float(method.rsplit("-", 1)[1])
        fn = progressive_ot4p if prog else ot4p
        r = fn(spec, a, b, tau=tau, seed=args.seed, device=args.device, log=log, power_device=args.ot4p_power_device)
        # As in the paper, model B is transformed by the final near-permutation matrices themselves.
        aligned = {k: v.cpu() for k, v in apply_matrices(spec, full_b, r.matrices, args.device).items()}
        return aligned, {"tau": tau, "seed": args.seed}
    if method in ("am", "ste"):
        if args.data is None:
            raise SystemExit(f"--data is required for {method}")
        from ppa.baselines.data_dependent import activation_matching, make_pipeline, ste_alignment

        torch.manual_seed(args.seed)
        pipeline = make_pipeline(args.data, args.device)
        if method == "am":
            perms = activation_matching(spec, pipeline, full_a, full_b, device=args.device)
        else:
            perms = ste_alignment(spec, pipeline, full_a, full_b, device=args.device, log=log)
        return apply_permutation(spec, full_b, {k: v.cpu() for k, v in perms.items()}), {"seed": args.seed}
    raise SystemExit(f"unknown method {method}")


def main() -> None:
    """Parse arguments, align and save."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--method", choices=METHODS, default="ours")
    parser.add_argument("--data", type=Path, default=None, help="scene directory (AM / STE only)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the stochastic baselines")
    parser.add_argument("--ot4p-power-device", default=None, help="see shuffle_recovery.py --help")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    full_a, full_b = load_state_dict(args.model_a), load_state_dict(args.model_b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    aligned, info = run(args.method, full_a, full_b, args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(aligned, args.out / "model_b_aligned.pth")
    a, b, ab = filter_nerf_params(full_a), filter_nerf_params(full_b), filter_nerf_params(aligned)
    summary = {
        "method": args.method,
        "device": args.device,
        "time_sec": elapsed,
        "l2_squared_before": squared_distance(a, b),
        "l2_squared_after": squared_distance(a, ab),
        "l1_before": l1_distance(a, b),
        "l1_after": l1_distance(a, ab),
        **info,
    }
    (args.out / "alignment.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
