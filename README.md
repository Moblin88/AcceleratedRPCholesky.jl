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

rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2.0)

# Returns n × rank matrix G and pivot indices
G, pivots = rpcholesky(rbf, X)  # Default rank: min(size(X, 1), 50)
approx = G * G'  # Low-rank approximation

# Or specify rank explicitly
G, pivots = rpcholesky(rbf, X, 30)
```

## API

### `rpcholesky(kernel, data, rank=min(n, 50); atol=1e-8, block_size=clamp(round(Int, √n), 4, 24))`

Compute a low-rank factor for the kernel matrix induced by `kernel` on rows of
`data`. `data` can be any one-based row-indexable collection supporting
`size(data, 1)` and `view(data, i, :)`, including matrices and DataFrames.

| Parameter    | Default              | Description                                           |
|--------------|----------------------|-------------------------------------------------------|
| `kernel`     | —                    | Callable `kernel(xᵢ, xⱼ) -> scalar`.                 |
| `data`       | —                    | `n × d` row-indexable data; rows are data points.     |
| `rank`       | `min(n, 50)`         | Optional positional maximum rank.                     |
| `atol`       | `1e-8`               | Stop when residual trace is at most this value.       |
| `block_size` | `clamp(round(Int, √n), 4, 24)` | Number of pivot candidates sampled per iteration.     |

Returns a tuple `(G, pivots)` where:
- `G` is an `n × rank` matrix where the first `length(pivots)` columns contain
  the low-rank factor and remaining columns are zero (if the algorithm stopped
  early). The factor uses the kernel's scalar type, so a `Float32` kernel returns
  a `Matrix{Float32}` factor.
- `pivots` is an integer vector containing the global indices of the accepted pivots.
  Its length is at most `rank`.

For a DataFrame, the kernel receives each observation as a `DataFrameRow`:

```julia
using DataFrames

data = DataFrame(x=randn(500), y=randn(500), z=randn(500))
rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2)
G, pivots = rpcholesky(rbf, data)      # Default rank: min(size(data, 1), 50)
G, pivots = rpcholesky(rbf, data, 30)  # Specify rank
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
- The output factor `G` is pre-allocated to `n × rank`; zero-padding is used if the
  algorithm terminates early due to numerical rank deficiency.
- Allocations are minimised via `view()` and in-place `mul!` / broadcast updates.

## License

MIT

---

> **Disclaimer:** This package was generated with the assistance of AI tools and reviewed by a human author.
