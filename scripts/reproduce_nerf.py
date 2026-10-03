"""Run alignment + LMC evaluation for all scenes and methods of the NeRF experiments.

Completed runs (``alignment.json`` / ``lmc/lmc.json`` present) are skipped, so the script can be
interrupted and restarted.

Example (paper Table 3 and supplementary Tables 1-3, barrier point only):
    python scripts/reproduce_nerf.py --data-root data/nerf_synthetic --weights weights --results results \
        --lambdas 0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["lego", "chair", "ficus", "hotdog", "mic", "ship", "materials", "drums"]
TABLE3 = [
    "naive", "wm", "prog-wm",
    "ot4p-0.3", "ot4p-0.5", "ot4p-0.7", "prog-ot4p-0.3", "prog-ot4p-0.5", "prog-ot4p-0.7",
    "ours-no-ref", "ours",
]  # fmt: skip
SUPP = [
    "am",
    "ste",
    "ours-no-svd-no-ref",
    "ours-no-svd",
    "ours-tau2-no-ref",
    "ours-tau2",
    "ours-tau5-no-ref",
    "ours-tau5",
]


def main() -> None:
    """Parse arguments and run every missing (scene, method) combination."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="NeRF Synthetic root (contains lego/, ...)")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--methods", nargs="+", default=TABLE3 + SUPP)
    parser.add_argument("--lambdas", type=float, nargs="*", default=None, help="default: 0, 0.1, ..., 1")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    for scene in args.scenes:
        for method in args.methods:
            out = args.results / scene / method
            wa, wb = args.weights / scene / "model_a.pth", args.weights / scene / "model_b.pth"
            if not (out / "alignment.json").exists():
                cmd = [py, str(ROOT / "scripts/align_nerf.py"), "--model-a", str(wa), "--model-b", str(wb),
                       "--method", method, "--out", str(out), "--data", str(args.data_root / scene)]  # fmt: skip
                print(f"[align] {scene} {method}", flush=True)
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            if not (out / "lmc" / "lmc.json").exists():
                cmd = [py, str(ROOT / "scripts/eval_lmc.py"), "--data", str(args.data_root / scene)]
                cmd += ["--model-a", str(wa), "--model-b", str(out / "model_b_aligned.pth")]
                cmd += ["--out", str(out / "lmc")]
                if args.lambdas:
                    cmd += ["--lambdas", *map(str, args.lambdas)]
                if args.no_png:
                    cmd.append("--no-png")
                print(f"[eval]  {scene} {method}", flush=True)
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
