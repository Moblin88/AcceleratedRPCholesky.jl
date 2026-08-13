# RPCholesky.jl

[![CI](https://github.com/Moblin88/RPCholesky.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Moblin88/RPCholesky.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/Moblin88/RPCholesky.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Moblin88/RPCholesky.jl)

Julia package implementing the **Accelerated Randomly Pivoted Cholesky** (RPCholesky)
algorithm for efficient low-rank approximation of positive semidefinite (PSD) matrices
and kernel matrices.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Moblin88/RPCholesky.jl")
```

## Quick Start

### Matrix interface

```julia
using RPCholesky, LinearAlgebra

n = 200
A_raw = randn(n, n)
A = A_raw * A_raw' + n * I   # symmetric PSD

# Approximate with rank ≤ 20, stopping when residual trace < 5% of original
F = rpcholesky(A; rank=20, rtol=0.05)   # returns CholeskyPivoted
L = F.factors[:, 1:F.rank]
println("Approximation error: ", norm(A - L*L') / norm(A))
```

### Kernel interface

```julia
using RPCholesky

X = randn(500, 4)   # 500 observations, 4 features

rbf(x, y) = exp(-sum((x .- y).^2) / 2.0)

# Returns n × k lower factor L such that L*L' ≈ K (kernel matrix)
L = rpcholesky_kernel(rbf, X; rank=30, rtol=0.05)
```

## API

### `rpcholesky(A; rank=nothing, rtol=0.05, block_size=nothing)`

Compute a low-rank Cholesky factor for a symmetric PSD matrix `A`.

| Parameter    | Default    | Description                                              |
|-------------|------------|----------------------------------------------------------|
| `rank`       | `nothing`  | Maximum rank. `nothing` means up to `n`.                 |
| `rtol`       | `0.05`     | Stop when residual trace ≤ `rtol × tr(A)`.              |
| `block_size` | `√n`       | Number of pivots sampled per block iteration.            |

Returns a `CholeskyPivoted` object `F`; use `F.factors[:, 1:F.rank]` for the `n × k` low-rank factor `L` such that `L * L' ≈ A`.

### `rpcholesky_kernel(kernel, X; rank=nothing, rtol=0.05, block_size=nothing)`

Compute a low-rank Cholesky factor for the kernel matrix induced by `kernel` on rows of `X`.

| Parameter    | Default    | Description                                              |
|-------------|------------|----------------------------------------------------------|
| `kernel`     | —          | Callable `kernel(xᵢ, xⱼ) -> scalar`.                    |
| `X`          | —          | `n × d` matrix; rows are data points.                    |
| `rank`       | `nothing`  | Maximum rank. `nothing` means dynamic (auto-sized).      |
| `rtol`       | `0.05`     | Stop when residual trace ≤ `rtol × tr(K)`.              |
| `block_size` | `√n`       | Number of pivots sampled per block iteration.            |

Returns an `n × k` matrix `L` such that `L * L'` approximates the kernel matrix.

## Algorithm

The implementation follows the **Accelerated RPCholesky** algorithm from:

> Epperly, E. N., Frangella, Z., Tropp, J. A., Webber, R. J., & Zangrando, M. (2024).
> *Accelerated Randomly Pivoted Cholesky.*
> [arXiv:2410.03969](https://arxiv.org/abs/2410.03969)

Key design choices:
- Pivots are sampled proportionally to the **residual diagonal** using `StatsBase.sample`
  with `Weights`, which is the RPCholesky selection rule.
- Processing is done in **blocks** to amortise sampling overhead.
- Allocations are minimised via `@views` and in-place `mul!` / broadcast updates.
- When `rank=nothing`, the output factor grows dynamically (doubling strategy).

## License

MIT
