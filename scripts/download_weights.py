"""Download the trained NeRF pairs of the paper from the Hugging Face Hub into ``weights/``.

Requires ``pip install huggingface_hub``.

Example:
    python scripts/download_weights.py              # all 8 scenes
    python scripts/download_weights.py --scenes lego
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "yuuxu/progressive-permutation-alignment"
SCENES = ["lego", "chair", "ficus", "hotdog", "mic", "ship", "materials", "drums"]
FILES = ["model_a.pth", "model_b.pth", "ref_perms_cpu.pt"]


def main() -> None:
    """Parse arguments and download the files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "weights")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub") from err

    for scene in args.scenes:
        for name in FILES:
            path = hf_hub_download(repo_id=args.repo_id, filename=f"{scene}/{name}", local_dir=args.out)
            print(path)


if __name__ == "__main__":
    main()
