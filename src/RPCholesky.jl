"""
    RPCholesky

Julia package implementing the Accelerated Randomly Pivoted Cholesky (RPCholesky)
algorithm for low-rank approximation of positive semidefinite (PSD) matrices and
kernel matrices.

The low-rank factor `G` satisfies `G * G' ≈ A` (matrix) or `G * G' ≈ K` (kernel).

# References
Chen, Y., Epperly, E. N., Tropp, J. A., & Webber, R. J. (2024).
*Randomly pivoted Cholesky: Practical approximation of a kernel matrix with
few entry evaluations.*
[arXiv:2207.06503](https://arxiv.org/abs/2207.06503)

Epperly, E. N., Frangella, Z., Tropp, J. A., Webber, R. J., & Zangrando, M. (2024).
*Accelerated Randomly Pivoted Cholesky.*
[arXiv:2410.03969](https://arxiv.org/abs/2410.03969)
"""
module RPCholesky

using LinearAlgebra
using StatsBase: sample, Weights

export rpcholesky, rpcholesky_kernel

# --- Internal helpers ---------------------------------------------------------

"""
    _default_block_size(n)

Heuristic block size: roughly sqrt(n), clipped to [4, 256].
"""
_default_block_size(n::Int) = clamp(round(Int, sqrt(n)), 4, 256)

"""
    _weighted_sample_block(d, b)

Draw up to `b` distinct indices from `1:length(d)` with probability proportional
to the residual diagonal `d` using `StatsBase.sample` with `Weights`.
Returns fewer than `b` indices when the effective support of `d` is small.
"""
function _weighted_sample_block(d::AbstractVector{T}, b::Int) where {T<:Real}
    w = max.(d, zero(T))
    total = sum(w)
    total <= zero(T) && return Int[]
    nnz = count(>(zero(T)), w)
    b = min(b, nnz)
    b == 0 && return Int[]
    return sample(1:length(d), Weights(w), b; replace=false)
end

# --- Core algorithm -----------------------------------------------------------

"""
    _rpcholesky_core!(G, piv, d, get_col!, n, max_cols, rtol, block_size) -> k

In-place Accelerated RPCholesky driver.  Fills up to `max_cols` columns of
the pre-allocated factor matrix `G` (n x max_cols) and records selected pivot
indices in `piv`.  Returns the number of columns `k` actually written.
"""
function _rpcholesky_core!(
    G::Matrix{T},
    piv::Vector{Int},
    d::Vector{T},
    get_col!::F,
    n::Int,
    max_cols::Int,
    rtol::Real,
    block_size::Int,
) where {T<:Real, F}

    trace0 = sum(d)
    trace0 <= zero(T) && return 0

    col_buf = Vector{T}(undef, n)
    k = 0

    while k < max_cols
        sum(d) <= rtol * trace0 && break

        b = min(block_size, max_cols - k)
        pivots = _weighted_sample_block(d, b)
        isempty(pivots) && break

        for pivot in pivots
            k >= max_cols && break

            get_col!(col_buf, pivot)

            if k > 0
                mul!(col_buf, view(G, :, 1:k), view(G, pivot, 1:k), -one(T), one(T))
            end

            diag_val = max(col_buf[pivot], zero(T))
            diag_val < eps(T) * trace0 && continue

            scale = one(T) / sqrt(diag_val)
            k += 1
            piv[k] = pivot

            view(G, :, k) .= col_buf .* scale

            Gk = view(G, :, k)
            @. d = max(d - Gk * Gk, zero(T))
        end
    end

    return k
end

# Variant supporting dynamic resizing of G when rank is not specified upfront.
function _rpcholesky_core_dynamic!(
    G_vec::Vector{Matrix{T}},
    piv_vec::Vector{Vector{Int}},
    d::Vector{T},
    get_col!::F,
    n::Int,
    max_cols::Int,
    rtol::Real,
    block_size::Int,
) where {T<:Real, F}

    trace0 = sum(d)
    trace0 <= zero(T) && return 0

    col_buf = Vector{T}(undef, n)
    k = 0

    while k < max_cols
        sum(d) <= rtol * trace0 && break

        b = min(block_size, max_cols - k)
        pivots = _weighted_sample_block(d, b)
        isempty(pivots) && break

        for pivot in pivots
            k >= max_cols && break

            get_col!(col_buf, pivot)

            G = G_vec[1]
            if k > 0
                mul!(col_buf, view(G, :, 1:k), view(G, pivot, 1:k), -one(T), one(T))
            end

            diag_val = max(col_buf[pivot], zero(T))
            diag_val < eps(T) * trace0 && continue

            scale = one(T) / sqrt(diag_val)
            k += 1

            if k > size(G, 2)
                new_cols = min(size(G, 2) * 2, n)
                G_new = Matrix{T}(undef, n, new_cols)
                G_new[:, 1:k-1] .= view(G, :, 1:k-1)
                G_vec[1] = G_new

                piv_new = Vector{Int}(undef, new_cols)
                piv_new[1:k-1] .= piv_vec[1][1:k-1]
                piv_vec[1] = piv_new
            end

            G = G_vec[1]
            piv_vec[1][k] = pivot
            view(G, :, k) .= col_buf .* scale

            Gk = view(G, :, k)
            @. d = max(d - Gk * Gk, zero(T))
        end
    end

    return k
end

# --- Public API ---------------------------------------------------------------

"""
    rpcholesky(A; rank=nothing, rtol=0.05, block_size=nothing) -> CholeskyPivoted

Compute an Accelerated Randomly Pivoted Cholesky low-rank approximation of the
`n x n` symmetric positive semidefinite matrix `A`.

Returns a `CholeskyPivoted` object `F` where:
- `F.rank` is the number of columns `k` of the computed factor.
- `F.factors[:, 1:F.rank]` is the `n x k` low-rank factor `L` with `L * L' ≈ A`.
- `F.p[1:F.rank]` contains the selected pivot indices.

The algorithm stops when:
1. `k == rank` (if `rank` is provided), **or**
2. `tr(residual) ≤ rtol × tr(A)`.

# Arguments
- `A`           : `n x n` symmetric PSD matrix.
- `rank`        : maximum rank (default: `n`).
- `rtol`        : relative trace tolerance (default: `0.05`).
- `block_size`  : pivots per block; defaults to `clamp(round(Int, sqrt(n)), 4, 256)`.

# Example
```julia
F = rpcholesky(A; rank=20, rtol=0.05)
L = F.factors[:, 1:F.rank]   # n x k low-rank factor
approx = L * L'               # approximately equal to A
```

# References
Epperly et al. (2024), *Accelerated Randomly Pivoted Cholesky*, arXiv:2410.03969.
"""
function rpcholesky(
    A::AbstractMatrix{T};
    rank::Union{Int,Nothing} = nothing,
    rtol::Real = 0.05,
    block_size::Union{Int,Nothing} = nothing,
) where {T<:Real}
    n = LinearAlgebra.checksquare(A)
    max_rank = isnothing(rank) ? n : min(rank, n)
    bs = isnothing(block_size) ? _default_block_size(n) : block_size

    G   = Matrix{T}(undef, n, max_rank)
    piv = Vector{Int}(undef, max_rank)
    d   = T[A[i,i] for i in 1:n]

    function get_col!(buf::Vector{T}, j::Int)
        @inbounds for i in 1:n
            buf[i] = A[i, j]
        end
    end

    k = _rpcholesky_core!(G, piv, d, get_col!, n, max_rank, T(rtol), bs)

    # Embed the n x k factor in an n x n matrix so that CholeskyPivoted can store
    # rank / pivot metadata.  The factor is NOT lower-triangular in general
    # (RPCholesky uses random pivots), so access the factor via
    # F.factors[:, 1:F.rank] rather than F.L.
    factors = zeros(T, n, n)
    if k > 0
        factors[:, 1:k] .= view(G, :, 1:k)
    end

    jpvt = vcat(piv[1:k], setdiff(1:n, view(piv, 1:k)))
    return CholeskyPivoted{T, Matrix{T}, Vector{Int}}(
        factors, 'L', jpvt, k, Float64(rtol), 0)
end

"""
    rpcholesky_kernel(kernel, X; rank=nothing, rtol=0.05, block_size=nothing) -> Matrix

Compute an Accelerated Randomly Pivoted Cholesky low-rank factor `L` such that
`L * L' ≈ K`, where `K[i,j] = kernel(X[i,:], X[j,:])`.

Only `O(n * k)` kernel evaluations are performed (vs `O(n^2)` for the full matrix).

# Arguments
- `kernel`      : callable `(xᵢ, xⱼ) -> scalar`.
- `X`           : `n x d` data matrix (rows are observations).
- `rank`        : maximum rank (default: `n`, dynamically allocated).
- `rtol`        : relative trace tolerance (default: `0.05`).
- `block_size`  : pivots per block; defaults to `clamp(round(Int, sqrt(n)), 4, 256)`.

# Returns
An `n x k` matrix `L` such that `L * L' ≈ K`.

# Example
```julia
rbf(x, y) = exp(-sum((x .- y).^2) / 2)
L = rpcholesky_kernel(rbf, X; rank=30, rtol=0.05)
approx = L * L'
```

# References
Epperly et al. (2024), *Accelerated Randomly Pivoted Cholesky*, arXiv:2410.03969.
"""
function rpcholesky_kernel(
    kernel::KF,
    X::AbstractMatrix{T};
    rank::Union{Int,Nothing} = nothing,
    rtol::Real = 0.05,
    block_size::Union{Int,Nothing} = nothing,
) where {KF, T<:Real}
    n = size(X, 1)
    max_rank = isnothing(rank) ? n : min(rank, n)
    bs = isnothing(block_size) ? _default_block_size(n) : block_size

    d = Vector{Float64}(undef, n)
    for i in 1:n
        xi = view(X, i, :)
        d[i] = Float64(kernel(xi, xi))
    end

    function get_col!(buf::Vector{Float64}, j::Int)
        xj = view(X, j, :)
        @inbounds for i in 1:n
            buf[i] = Float64(kernel(view(X, i, :), xj))
        end
    end

    if isnothing(rank)
        init_cols = min(_default_block_size(n) * 2, n)
        G_vec   = [Matrix{Float64}(undef, n, init_cols)]
        piv_vec = [Vector{Int}(undef, init_cols)]
        k = _rpcholesky_core_dynamic!(
            G_vec, piv_vec, d, get_col!, n, max_rank, Float64(rtol), bs)
        return G_vec[1][:, 1:k]
    else
        G   = Matrix{Float64}(undef, n, max_rank)
        piv = Vector{Int}(undef, max_rank)
        k = _rpcholesky_core!(G, piv, d, get_col!, n, max_rank, T(rtol), bs)
        return G[:, 1:k]
    end
end

end # module RPCholesky
