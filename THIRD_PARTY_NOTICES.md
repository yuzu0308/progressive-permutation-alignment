# Third-party notices

## OT4P

`ppa/spectral.py` (and the OT4P baseline in `ppa/baselines/`) contain code adapted from
OT4P, https://github.com/YamingGuo98/OT4P (commit 99915004, 2024-11-08).

Modifications:

- The matrix power in `orth2perm` is computed with an SVD, `U diag(S^tau) V^T`, instead of an
  eigendecomposition.
- The powered matrix is anchored to the identity, `tau * M^<tau> + (1 - tau) I`, before being
  multiplied by the Hungarian anchor.
- The batched linear assignment uses `scipy.optimize.linear_sum_assignment` instead of
  `torch_linear_assignment`.

Note that the "OT4P" baselines reported in the paper use this modified `orth2perm`.

```
MIT License

Copyright (c) 2024 Yaming Guo 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## nerfstudio

`ppa/render.py` re-implements the forward pass of nerfstudio's `vanilla-nerf` model
(https://github.com/nerfstudio-project/nerfstudio, Apache License 2.0,
Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors).
