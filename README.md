# AcceleratedRPCholesky.jl

[![CI](https://github.com/Moblin88/AcceleratedRPCholesky.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Moblin88/AcceleratedRPCholesky.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/Moblin88/AcceleratedRPCholesky.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Moblin88/AcceleratedRPCholesky.jl)

Julia package implementing the **Accelerated Randomly Pivoted Cholesky** (RPCholesky)
algorithm for efficient low-rank approximation of kernel matrices.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Moblin88/AcceleratedRPCholesky.jl")
```

## Quick Start

```julia
using AcceleratedRPCholesky

X = randn(500, 4)   # 500 observations, 4 features

rbf(x, y) = exp(-sum((x .- y).^2) / 2.0)

# Returns n × k factor G such that G*G' ≈ K (kernel matrix)
G = rpcholesky(rbf, X; rank=30, rtol=0.05)
approx = G * G'
```

## API

### `rpcholesky(kernel, X; rank=n, rtol=0.05, block_size=clamp(round(Int, √n), 4, 256))`

Compute a low-rank factor for the kernel matrix induced by `kernel` on rows of `X`.

| Parameter    | Default              | Description                                           |
|--------------|----------------------|-------------------------------------------------------|
| `kernel`     | —                    | Callable `kernel(xᵢ, xⱼ) -> scalar`.                 |
| `X`          | —                    | `n × d` matrix; rows are data points.                 |
| `rank`       | `n`                  | Maximum rank.                                         |
| `rtol`       | `0.05`               | Stop when residual trace ≤ `rtol × tr(K)`.           |
| `block_size` | `clamp(round(Int, √n), 4, 256)` | Number of pivot candidates sampled per iteration.     |

Returns an `n × k` matrix `G` such that `G * G'` approximates the kernel matrix.

## Algorithm

The implementation follows the **Accelerated RPCholesky** algorithm from:

> Epperly, E. N., Frangella, Z., Tropp, J. A., Webber, R. J., & Zangrando, M. (2024).
> *Accelerated Randomly Pivoted Cholesky.*
> [arXiv:2410.03969](https://arxiv.org/abs/2410.03969)

Key design choices:
- Pivots are sampled proportionally to the **residual diagonal** using `StatsBase.sample`
  with `Weights`, which is the RPCholesky selection rule.
- Processing is done in **blocks** to amortise sampling overhead; within each block,
  acceptance-rejection (Algorithm 2.1) decides which candidates are accepted cheaply
  before evaluating full kernel columns.
- The output factor `G` grows in multiples of `block_size` as pivots are accepted.
- Allocations are minimised via `view()` and in-place `mul!` / broadcast updates.

## License

MIT

---

> **Disclaimer:** This package was generated with the assistance of AI tools and reviewed by a human author.
