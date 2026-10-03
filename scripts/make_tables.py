"""Write the shuffle-recovery and timing tables of the paper as LaTeX from the benchmark results.

Tables (written to ``--out``/tables.tex):
    table1_random      Table 1 with randomly initialised MLP5 / VGG11 / ResNet18 (as in the paper's runs)
    table1_pretrained  Table 1 with ImageNet weights for VGG11 / ResNet18 (MLP5 stays random)
    table2             Table 2, NeRF shuffle recovery (8 scenes)
    supp_table4_*      supp. Table 4, alignment time for the two Table 1 variants
    supp_table5        supp. Table 5, NeRF alignment time (shuffle / independently trained pairs)

Missing entries are written as "--". Entries that hit the time limit are wrapped in parentheses.

Example:
    python scripts/make_tables.py --results results --out tables
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROWS: List[Tuple[str, str]] = [
    ("wm", "Weight Matching"),
    ("prog-wm", "Progressive Weight Matching"),
    ("ot4p-0.7", r"OT4P ($\tau=0.7$)"),
    ("ot4p-0.5", r"OT4P ($\tau=0.5$)"),
    ("ot4p-0.3", r"OT4P ($\tau=0.3$)"),
    ("prog-ot4p-0.7", r"Progressive OT4P ($\tau=0.7$)"),
    ("prog-ot4p-0.5", r"Progressive OT4P ($\tau=0.5$)"),
    ("prog-ot4p-0.3", r"Progressive OT4P ($\tau=0.3$)"),
    ("ours-no-ref", "Proposed (Ours) w/o Ref."),
    ("ours", "Proposed (Ours)"),
]
SCENES = ["lego", "chair", "ficus", "hotdog", "mic", "ship", "materials", "drums"]
TIMEOUT = 3600.0
MARK_TIMEOUTS = True  # set by --no-timeout-marks
NOTE_PRETRAINED = (
    ". VGG11 and ResNet18 use ImageNet weights; MLP5 is randomly initialised. "
    "The proposed method is computed in float64 on ResNet18. One VGG11 run of OT4P ($\\tau=0.5$) was computed on "
    "the CPU because of a GPU error."
)


def sig(x: float, digits: int = 4) -> str:
    """Format with ``digits`` significant digits as in the paper (``0.000`` for zero)."""
    if x == 0 or not math.isfinite(x):
        return "0.000" if x == 0 else "--"
    decimals = max(0, digits - 1 - int(math.floor(math.log10(abs(x)))))
    return f"{x:.{decimals}f}"


def stat(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def cell(
    values: Optional[Sequence[float]],
    timed_out: bool = False,
    with_std: bool = True,
    std_decimals: Optional[int] = None,
) -> str:
    """``mean ±std`` (or ``--``), parenthesised if a run hit the time limit.

    A single run is written without a standard deviation and marked with a dagger. ``std_decimals`` fixes the
    decimals of the standard deviation (the paper uses 3 for L1 / precision and 4 significant digits for times).
    """
    if not values:
        return "--"
    m, s = stat(values)
    if with_std and len(values) == 1:
        text = f"{sig(m)}$^\\dagger$"
    elif with_std:
        std = f"{s:.{std_decimals}f}" if std_decimals is not None else sig(s)
        text = f"{sig(m)} $\\pm${std}"
    else:
        text = sig(m)
    return f"({text})" if timed_out else text


class Records:
    """Shuffle-recovery records of several result files, keyed by (model, method)."""

    def __init__(self, paths: Sequence[Path]) -> None:
        """Load and merge the JSON files that exist; later files override earlier ones per (model, method)."""
        self.by_key: Dict[Tuple[str, str], List[dict]] = {}
        for path in paths:
            if not path.exists():
                continue
            grouped: Dict[Tuple[str, str], List[dict]] = {}
            for r in json.loads(path.read_text()):
                grouped.setdefault((r["model"], r["method"]), []).append(r)
            self.by_key.update(grouped)

    def get(self, model: str, method: str) -> List[dict]:
        """Records of one model / method (possibly empty)."""
        return self.by_key.get((model, method), [])


def quality_table(recs: Records, models: Sequence[Tuple[str, str]], caption: str, label: str) -> str:
    """LaTeX for a Table-1-style table (log(1+L1) and precision per model)."""
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for _, name in models)
    sub = " & ".join(["L1 Dist. & Precision"] * len(models))
    lines = [
        "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{l" + "cc" * len(models) + "}", "\\toprule",
        f"Algorithm & {head} \\\\", f" & {sub} \\\\", "\\midrule",
    ]  # fmt: skip
    for method, name in ROWS:
        cells = []
        for model, _ in models:
            rs = recs.get(model, method)
            out = MARK_TIMEOUTS and any(r.get("timed_out") for r in rs)
            cells.append(cell([math.log1p(r["l1"]) for r in rs], out, std_decimals=3))
            cells.append(cell([100 * r["precision"] for r in rs], out, std_decimals=3))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def time_table(recs: Records, models: Sequence[Tuple[str, str]], caption: str, label: str) -> str:
    """LaTeX for a timing table (mean ±std seconds per model)."""
    lines = [
        "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(models) + "}", "\\toprule",
        "Algorithm & " + " & ".join(name for _, name in models) + " \\\\", "\\midrule",
    ]  # fmt: skip
    for method, name in ROWS:
        cells = []
        for model, _ in models:
            rs = recs.get(model, method)
            cells.append(cell([r["time_sec"] for r in rs], MARK_TIMEOUTS and any(r.get("timed_out") for r in rs)))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def nerf_table(recs: Records) -> str:
    """LaTeX for Table 2 (NeRF shuffle recovery, mean over scenes, no std as in the paper)."""
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{L1 distance (converted by $\\log(1+x)$) and Precision (\\%) on NeRF}", "\\label{tab:nerf_shuffle}",
        "\\begin{tabular}{lcc}", "\\toprule", "Algorithm & L1 Dist. & Precision \\\\", "\\midrule",
    ]  # fmt: skip
    for method, name in ROWS:
        rs = recs.get("nerf", method)
        l1 = cell([math.log1p(r["l1"]) for r in rs], with_std=False)
        pr = cell([100 * r["precision"] for r in rs], with_std=False)
        lines.append(f"{name} & {l1} & {pr} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def nerf_time_table(recs: Records, results: Path) -> str:
    """LaTeX for supp. Table 5: NeRF alignment time for shuffled copies and independently trained pairs."""
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Alignment computation time on NeRF (mean and standard deviation over 8 scenes, [sec])}",
        "\\label{tab:nerf_time}", "\\begin{tabular}{lcc}", "\\toprule",
        "Algorithm & Shuffled copy & Independently trained \\\\", "\\midrule",
    ]  # fmt: skip
    for method, name in ROWS:
        shuffle = [r["time_sec"] for r in recs.get("nerf", method)]
        pairs = []
        for scene in SCENES:
            p = results / scene / method / "alignment.json"
            if p.exists():
                pairs.append(json.loads(p.read_text())["time_sec"])
        lines.append(f"{name} & {cell(shuffle)} & {cell(pairs)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main() -> None:
    """Parse arguments and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("tables"))
    parser.add_argument("--no-timeout-marks", action="store_true", help="do not parenthesise timed-out entries")
    args = parser.parse_args()
    global MARK_TIMEOUTS
    MARK_TIMEOUTS = not args.no_timeout_marks
    args.out.mkdir(parents=True, exist_ok=True)

    shuffle = Records([args.results / "shuffle" / "shuffle_recovery.json"])
    pretrained = Records(
        [
            args.results / "shuffle" / "shuffle_recovery.json",
            args.results / "shuffle_pretrained" / "shuffle_recovery.json",
            args.results / "shuffle_pretrained_fp64" / "shuffle_recovery.json",
        ]
    )
    random_models = [("mlp5", "MLP5"), ("vgg11", "VGG11"), ("resnet18", "ResNet18")]
    pre_models = [("mlp5", "MLP5"), ("vgg11-pretrained", "VGG11"), ("resnet18-pretrained", "ResNet18")]
    cap1 = "L1 distance (converted by $\\log(1+x)$) and Precision (\\%) of alignment algorithms across standard architectures (MLP, CNN)"  # noqa: E501
    tables = {
        "table1_random": quality_table(shuffle, random_models, cap1, "tab:shuffle"),
        "table1_pretrained": quality_table(pretrained, pre_models, cap1 + NOTE_PRETRAINED, "tab:shuffle"),
        "table2": nerf_table(shuffle),
        "supp_table4_random": time_table(
            shuffle, random_models, "Alignment computation time (mean and standard deviation, [sec])", "tab:time"
        ),
        "supp_table4_pretrained": time_table(
            pretrained,
            pre_models,
            "Alignment computation time (mean and standard deviation, [sec])" + NOTE_PRETRAINED,
            "tab:time",
        ),
        "supp_table5": nerf_time_table(shuffle, args.results),
    }
    tex = "\n\n".join(f"% ---- {name} ----\n{body}" for name, body in tables.items())
    (args.out / "tables.tex").write_text(tex + "\n", encoding="utf-8")
    print(args.out / "tables.tex")


if __name__ == "__main__":
    main()
