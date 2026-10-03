"""Evaluate linear mode connectivity of a NeRF pair on NeRF Synthetic test views.

For every lambda in 0, 0.1, ..., 1 the weights ``(1 - lambda) A + lambda B`` are rendered on the
10 test views of the paper (``sorted(RandomState(42).choice(200, 10))``) and PSNR / RMSE / SSIM
against the ground truth (composited on white) are computed on the float renders (``metrics_float``).
The renders are also written to disk as PNGs.

Example:
    python scripts/eval_lmc.py --data data/nerf_synthetic/lego --model-a weights/lego/model_a.pth \
        --model-b results/lego/ours/model_b_aligned.pth --out results/lego/ours/lmc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppa import interpolate  # noqa: E402
from ppa import render as R  # noqa: E402
from ppa.metrics import ssim  # noqa: E402
from ppa.nerf import load_state_dict  # noqa: E402


def main() -> None:
    """Parse arguments, render and write ``lmc.json``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="NeRF Synthetic scene directory")
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True, help="(aligned) model B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lambdas", type=float, nargs="*", default=None, help="default: 0, 0.1, ..., 1")
    parser.add_argument("--no-png", action="store_true", help="do not keep the rendered PNGs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    lams = args.lambdas if args.lambdas else list(R.lambdas())
    a, b = load_state_dict(args.model_a), load_state_dict(args.model_b)
    views = R.load_test_views(args.data)
    gts = {idx: R.ground_truth(views, idx).to(args.device) for idx in views.indices}

    result: Dict[str, object] = {"views": views.indices, "lambdas": lams, "metrics_float": {}}
    t0 = time.perf_counter()
    for lam in lams:
        model = R.build_model(interpolate(a, b, lam), views.scene_box, args.device)
        per: Dict[str, List[float]] = {"psnr": [], "rmse": [], "ssim": []}
        lam_dir = args.out / f"lambda_{lam:.1f}"
        if not args.no_png:
            lam_dir.mkdir(parents=True, exist_ok=True)
        for i, idx in enumerate(views.indices):
            img = R.render_view(model, views, idx, args.device)
            if not args.no_png:
                Image.fromarray(R.to_uint8(img)).save(lam_dir / f"rendered_{i:04d}_idx{idx:04d}.png")
            per["psnr"].append(R.psnr(img, gts[idx]))
            per["rmse"].append(R.rmse(img, gts[idx]))
            per["ssim"].append(ssim(img, gts[idx]))
        result["metrics_float"][f"{lam:.1f}"] = per  # type: ignore[index]
        f = {m: float(np.mean(v)) for m, v in per.items()}
        print(f"lambda={lam:.1f} PSNR={f['psnr']:.3f} RMSE={f['rmse']:.4f} SSIM={f['ssim']:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["render_time_sec"] = time.perf_counter() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lmc.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
