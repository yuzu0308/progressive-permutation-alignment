"""Aggregate LMC results over scenes into the tables and curves of the paper.

Reads ``<results>/<scene>/<method>/lmc/lmc.json`` written by ``eval_lmc.py`` and prints, for every
method, the mean over scenes (and over the 10 views per scene) of RMSE / PSNR / SSIM at the
barrier point ``lambda = 0.5`` (Table 3, supplementary Tables 1-3). With ``--plot`` the
metric-vs-lambda curves (Fig. 4) are written as well.

Example:
    python scripts/aggregate_lmc.py --results results --methods naive wm prog-wm ours-no-ref ours --plot figs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

SCENES = ["lego", "chair", "ficus", "hotdog", "mic", "ship", "materials", "drums"]
METRICS = ("rmse", "psnr", "ssim")


def load(results: Path, scene: str, method: str) -> Dict[str, Dict[str, List[float]]]:
    """Per-lambda metric lists of one scene/method.

    Args:
        results: Results root.
        scene: Scene name.
        method: Method name.

    Returns:
        ``{lambda: {"psnr": [...], "rmse": [...], "ssim": [...]}}``.
    """
    data = json.loads((results / scene / method / "lmc" / "lmc.json").read_text())
    return {lam: dict(m) for lam, m in data["metrics_float"].items()}


def main() -> None:
    """Parse arguments and print the aggregated table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--lam", default="0.5")
    parser.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    parser.add_argument("--plot", type=Path, default=None, help="directory for metric-vs-lambda plots")
    args = parser.parse_args()

    table: Dict[str, Dict[str, float]] = {}
    curves: Dict[str, Dict[str, List[float]]] = {}
    lams: List[str] = []
    for method in args.methods:
        present = [s for s in args.scenes if (args.results / s / method / "lmc" / "lmc.json").exists()]
        if len(present) != len(args.scenes):
            missing = sorted(set(args.scenes) - set(present))
            print(f"[warn] {method}: missing scenes {missing}; averaging over {len(present)}", file=sys.stderr)
        if not present:
            continue
        per_scene = [load(args.results, s, method) for s in present]
        lams = sorted(per_scene[0], key=float)
        table[method] = {m: float(np.mean([np.mean(d[args.lam][m]) for d in per_scene])) for m in METRICS}
        table[method]["num_scenes"] = len(present)
        curves[method] = {m: [float(np.mean([np.mean(d[lam][m]) for d in per_scene])) for lam in lams] for m in METRICS}

    print(f"lambda={args.lam}, scenes={len(args.scenes)}")
    print(f"{'method':24s} {'RMSE':>7s} {'PSNR':>7s} {'SSIM':>7s}  scenes")
    for method, row in table.items():
        print(f"{method:24s} {row['rmse']:7.3f} {row['psnr']:7.2f} {row['ssim']:7.3f}  {int(row['num_scenes'])}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"table": table, "lambdas": lams, "curves": curves}, indent=1))

    if args.plot is not None:
        import matplotlib.pyplot as plt

        args.plot.mkdir(parents=True, exist_ok=True)
        x = [float(v) for v in lams]
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for method, c in curves.items():
                ax.plot(x, c[metric], marker="o", ms=3, label=method)
            ax.set_xlabel("lambda")
            ax.set_ylabel(metric.upper())
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(args.plot / f"lmc_{metric}.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
