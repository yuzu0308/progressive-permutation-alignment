# Progressive Permutation Alignment with Temperature-Controlled Spectral Sharpening

**ACCV 2026** · Yuta Horioka\*, Naoya Chiba\*, Yuki Uranishi (The University of Osaka) · \*equal contribution

[Project page](https://yuzu0308.github.io/progressive-permutation-alignment/) ·
Paper (TODO) · Supplementary (TODO) · Weights (TODO: Hugging Face)

<p align="center"><img src="docs/static/img/overview.png" width="85%" alt="Method overview"></p>

Independently trained neural fields that represent the same scene differ by permutations of their hidden
units, so linearly interpolating their weights passes through a high loss barrier. This repository contains
the implementation of our alignment method and the code to reproduce every table of the paper and the
supplementary material.

The method visits the layers one by one. For each layer it computes the correlation matrix `C` with the
other layers' current (soft) permutations, takes its Hungarian solution `P̂` as an anchor, and updates the
layer to `P̂ [τ U Σ^τ Vᵀ + (1 − τ) I]` where `P̂ᵀC = U Σ Vᵀ`. The temperature is annealed over `τ ∈ {1, 0.5, 0}`,
after which a layer-wise weight-matching refinement starts from the result. There is no gradient descent.

## Installation

The alignment itself only needs PyTorch, NumPy and SciPy:

```bash
pip install -e .
```

Rendering-based evaluation (NeRF interpolation, activation matching, STE) needs nerfstudio. The paper
results were produced with Python 3.8, PyTorch 2.1.2 + CUDA 11.8 and nerfstudio 1.1.5:

```bash
conda create -n ppa python=3.8 -y && conda activate ppa
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[nerf,cnn,dev]"
```

## Quick start

```python
from ppa import align, apply_permutation, interpolate
from ppa.nerf import filter_nerf_params, load_state_dict, vanilla_nerf_permutation_spec

a = load_state_dict("weights/lego/model_a.pth")   # nerfstudio vanilla-nerf state dicts
b = load_state_dict("weights/lego/model_b.pth")
spec = vanilla_nerf_permutation_spec()

result = align(spec, filter_nerf_params(a), filter_nerf_params(b), device="cuda")
b_aligned = apply_permutation(spec, b, result.perms)
midpoint = interpolate(a, b_aligned, 0.5)
```

For other architectures, describe which weight axes share a permutation with a `PermutationSpec`
(see `ppa/spec.py` and `ppa/zoo.py` for an MLP, VGG11 and ResNet18).

## Data and trained models

* NeRF Synthetic: download [`nerf_synthetic`](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
  and place the eight scenes under `data/nerf_synthetic/`.
* Trained models: the 16 vanilla NeRFs used in the paper (8 scenes × seeds 42/100, 1M iterations) are
  available on Hugging Face (TODO). Put them under `weights/<scene>/model_{a,b}.pth`. The files
  `weights/<scene>/ref_perms_cpu.pt` hold the permutations produced by the original research code and are
  used by the regression test.
* To train them yourself:

  ```bash
  ns-train vanilla-nerf --machine.seed 42  --max-num-iterations 1000000 --data data/nerf_synthetic/lego blender-data
  ns-train vanilla-nerf --machine.seed 100 --max-num-iterations 1000000 --data data/nerf_synthetic/lego blender-data
  python scripts/export_weights.py --config outputs/lego/vanilla-nerf/<run>/config.yml --out weights/lego/model_a.pth
  ```

## Reproducing the paper

| Result | Command |
|---|---|
| Table 3, supp. Tables 1–3 (NeRF interpolation barrier) | `python scripts/reproduce_nerf.py --data-root data/nerf_synthetic --lambdas 0.5` then `python scripts/aggregate_lmc.py --methods naive wm prog-wm ours-no-ref ours` |
| Fig. 4 (metrics along λ) | `python scripts/reproduce_nerf.py --data-root data/nerf_synthetic --methods naive wm prog-wm ours-no-ref ours` then `aggregate_lmc.py ... --plot figs/` |
| Tables 1–2, supp. Tables 4–5 (shuffle recovery, time) | `python scripts/shuffle_recovery.py --models mlp5 resnet18 vgg11 nerf` |
| Single pair, any method | `python scripts/align_nerf.py --method ours --model-a ... --model-b ... --out ...` and `python scripts/eval_lmc.py ...` |

Method names: `ours`, `ours-no-ref`, the ablations `ours-no-svd[-no-ref]`, `ours-tau2[-no-ref]`,
`ours-tau5[-no-ref]`, and the baselines `naive`, `wm`, `prog-wm`, `ot4p-{0.3,0.5,0.7}`,
`prog-ot4p-{0.3,0.5,0.7}`, `am`, `ste`. Evaluating one method on all scenes at a single λ takes about
20 minutes on an RTX 5090, dominated by volume rendering.

## Notes on reproducibility

* The implementation reproduces the permutations of the code used for the paper bit-for-bit
  (`pytest tests/test_align.py`, needs the downloaded weights), and the rendered images of the proposed
  method and its ablations are pixel-identical to the ones behind the paper tables.
* Layer order: stage 1 visits the permutation groups in `sorted` order of their ids, which for the NeRF spec
  processes the colour-head groups before the trunk groups of each field. This is the order used for the
  paper; pass `order=` to `align` to change it.
* Correlations are normalised per layer (Frobenius norm) before summation, and each temperature runs up to
  100 sweeps until the alignment cost stops decreasing.
* As in OT4P, the last column of the anchor is sign-flipped when its determinant is negative, so at τ = 1
  the update equals `C` up to the sign of its last column.
* The OT4P baselines use OT4P's original `orth2perm` (eigendecomposition power, no anchoring to the identity),
  bit-identical to the upstream implementation; `variant="modified"` switches to the map of our method. They
  start from random parameters and the original scripts did not fix a seed; ours default to `--seed 0`, so
  their numbers differ slightly from the paper.
* Metrics: by default every method is evaluated the same way, on the float renders with torchmetrics' SSIM
  (Gaussian window). This reproduces all RMSE / PSNR values of the paper. A few table entries were computed
  differently in the paper; they can be obtained with the following options:

  ```bash
  # supp. Tables 2-3: SSIM of Naive / WM / Progressive WM (scikit-image SSIM on the saved PNGs)
  python scripts/aggregate_lmc.py --ssim skimage --methods naive wm prog-wm
  # supp. Table 3: "w/o Ref." rows of the two- and five-step schedules (averaged without lego)
  python scripts/aggregate_lmc.py --scenes chair ficus hotdog mic ship materials drums --methods ours-tau2-no-ref ours-tau5-no-ref
  ```

## Citation

```bibtex
@inproceedings{horioka2026progressive,
  title     = {Progressive Permutation Alignment with Temperature-Controlled Spectral Sharpening},
  author    = {Horioka, Yuta and Chiba, Naoya and Uranishi, Yuki},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## License and acknowledgements

MIT License. `ppa/spectral.py` and `ppa/baselines/ot4p.py` adapt code from
[OT4P](https://github.com/YamingGuo98/OT4P) (MIT), and `ppa/render.py` builds on
[nerfstudio](https://github.com/nerfstudio-project/nerfstudio) (Apache-2.0); see `THIRD_PARTY_NOTICES.md`.
Weight Matching follows [Git Re-Basin](https://github.com/samuela/git-re-basin).
This work was supported by JST PRESTO Grant Number JPMJPR25T6.
