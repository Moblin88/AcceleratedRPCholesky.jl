# AcceleratedRPCholesky.jl

[![CI](https://github.com/Moblin88/AcceleratedRPCholesky.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Moblin88/AcceleratedRPCholesky.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/Moblin88/AcceleratedRPCholesky.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Moblin88/AcceleratedRPCholesky.jl)

Julia package implementing the **Accelerated Randomly Pivoted Cholesky** (RPCholesky)
algorithm for efficient low-rank approximation of kernel matrices.
For data rows `xᵢ`, `K` denotes the `n × n` kernel matrix with
`K[i, j] = kernel(xᵢ, xⱼ)`.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Moblin88/AcceleratedRPCholesky.jl")
```

## Quick Start

```julia
using AcceleratedRPCholesky

X = randn(500, 4)   # 500 observations, 4 features

rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2.0)

# Returns a dynamically sized factor G
G = rpcholesky(rbf, X)  # Default rank: min(size(X, 1), 50)
approx = G * G'  # Low-rank approximation

# Or specify the maximum rank and relative trace tolerance
G = rpcholesky(rbf, X; rank=30, rtol=0.01)

# Request pivots explicitly
G, pivots = rpcholesky(rbf, X, Val(true); rank=30, rtol=0.01)
```

## API

### `rpcholesky(kernel, data; rank=min(n, 50), rtol=0.05, atol=1e-8, block_size=clamp(round(Int, √n), 4, 24), rng=Random.default_rng())`

Compute a low-rank factor for the kernel matrix induced by `kernel` on rows of
`data`. `data` can be any one-based row-indexable collection supporting
`size(data, 1)` and `view(data, i, :)`, including matrices and DataFrames.

| Parameter    | Default              | Description                                           |
|--------------|----------------------|-------------------------------------------------------|
| `kernel`     | —                    | Callable `kernel(xᵢ, xⱼ) -> scalar`.                 |
| `data`       | —                    | `n × d` row-indexable data; rows are data points.     |
| `rank`       | `min(n, 50)`         | Maximum rank.                                         |
| `rtol`       | `0.05`               | Stop when residual trace is at most `rtol * tr(K)`.   |
| `atol`       | `1e-8`               | Stop when residual trace is at most this value.       |
| `block_size` | `clamp(round(Int, √n), 4, 24)` | Number of pivot candidates sampled per iteration.     |
| `rng`        | `Random.default_rng()` | RNG for candidate and rejection sampling.            |

> **Memory note:** `rpcholesky` allocates storage for up to `rank` columns of
> `G` up front (`n × rank`), then truncates to the accepted rank on return.
> Use large `rank` values with care: this allocation is paid even when the
> algorithm terminates early due to `rtol` or `atol`.

Returns:
- `rpcholesky(kernel, data; ...)` returns `G`, an `n × k` matrix.
  The factor uses the kernel's scalar type, so a `Float32` kernel returns a
  `Matrix{Float32}` factor.
- `rpcholesky(kernel, data, Val(true); ...)` returns `(G, pivots)`, where
  `pivots` is an integer vector containing the global indices of the accepted pivots
  and `k = length(pivots)`.

For a DataFrame, the kernel receives each observation as a `DataFrameRow`:

```julia
using DataFrames

data = DataFrame(x=randn(500), y=randn(500), z=randn(500))
rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2)
G = rpcholesky(rbf, data)
G = rpcholesky(rbf, data; rank=30, rtol=0.01)
G, pivots = rpcholesky(rbf, data, Val(true); rank=30, rtol=0.01)
```

## Algorithm

The implementation follows the **Accelerated RPCholesky** algorithm from:

> Epperly, E. N., Frangella, Z., Tropp, J. A., Webber, R. J., & Zangrando, M. (2024).
> *Accelerated Randomly Pivoted Cholesky.*
> [arXiv:2410.03969](https://arxiv.org/abs/2410.03969)

Key design choices:
- Pivots are sampled proportionally to the **residual diagonal**, which is the
  RPCholesky selection rule.
- Processing is done in **blocks** to amortise sampling overhead; within each block,
  acceptance-rejection (Algorithm 2.1) decides which candidates are accepted cheaply
  before evaluating full kernel columns.
- The factor is dynamically sized to the accepted rank and is at most `n × rank`.
- Allocations are minimised via `view()` and in-place `mul!` / broadcast updates.

## License

MIT

---

> **Disclaimer:** This package was generated with the assistance of AI tools and reviewed by a human author.
