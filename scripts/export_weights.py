"""Export the model state dict of a trained nerfstudio ``vanilla-nerf`` run.

Example:
    python scripts/export_weights.py --config outputs/lego_seed42/vanilla-nerf/<run>/config.yml \
        --out weights/lego/model_a.pth

nerfstudio resolves the checkpoint directory relative to the working directory, so run this
from the directory where ``ns-train`` was run (or pass ``--workdir``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    """Parse arguments and export the state dict."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="config.yml written by ns-train")
    parser.add_argument("--out", type=Path, required=True, help="output .pth file")
    parser.add_argument("--workdir", type=Path, default=None, help="directory ns-train was run from")
    args = parser.parse_args()

    out = args.out.resolve()
    config = args.config.resolve() if args.workdir is None else args.config
    if args.workdir is not None:
        os.chdir(args.workdir)
    if not Path(config).exists():
        raise FileNotFoundError(f"config not found: {config}")

    from nerfstudio.utils.eval_utils import eval_setup

    cfg, pipeline, _, _ = eval_setup(Path(config))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pipeline.model.state_dict(), out)
    print(f"seed={cfg.machine.seed} -> {out}")


if __name__ == "__main__":
    main()
