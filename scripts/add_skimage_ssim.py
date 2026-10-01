"""Add scikit-image SSIM (computed on the saved PNGs) to existing ``lmc.json`` files.

``eval_lmc.py`` already records it for new runs; this back-fills results produced before the
option existed, without re-rendering.

Example:
    python scripts/add_skimage_ssim.py --results results --data-root data/nerf_synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppa import render as R  # noqa: E402
from ppa.metrics import ssim_skimage  # noqa: E402


def main() -> None:
    """Parse arguments and update every ``lmc.json`` that lacks ``ssim_skimage``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    gts = {}
    for path in sorted(args.results.glob("*/*/lmc/lmc.json")):
        scene = path.parts[-4]
        data = json.loads(path.read_text())
        pending = [lam for lam, m in data["metrics_png"].items() if "ssim_skimage" not in m]
        if not pending:
            continue
        if scene not in gts:
            views = R.load_test_views(args.data_root / scene)
            gts[scene] = {idx: R.ground_truth(views, idx).numpy() for idx in views.indices}
        for lam in pending:
            values = []
            for i, idx in enumerate(data["views"]):
                png = path.parent / f"lambda_{float(lam):.1f}" / f"rendered_{i:04d}_idx{idx:04d}.png"
                if not png.exists():
                    raise FileNotFoundError(f"{png} (re-run eval_lmc.py without --no-png)")
                pred = np.asarray(Image.open(png), dtype=np.float32) / 255.0
                values.append(ssim_skimage(pred, gts[scene][idx]))
            data["metrics_png"][lam]["ssim_skimage"] = values
        path.write_text(json.dumps(data, indent=1))
        print(f"{path.parts[-4]}/{path.parts[-3]}: {', '.join(pending)}")


if __name__ == "__main__":
    main()
