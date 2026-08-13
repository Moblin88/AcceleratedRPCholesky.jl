"""
    RPCholesky

Julia package implementing the Accelerated Randomly Pivoted Cholesky (RPCholesky)
algorithm for low-rank approximation of positive semidefinite (PSD) matrices and
kernel matrices.

The low-rank factor `G` (n × k) satisfies `G * G' ≈ A`.

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

# ---------------------------------------------------------------------------
# Algorithm 2.1 — RejectionCholesky
# ---------------------------------------------------------------------------

"""
    _rejection_cholesky!(H) -> (L, accepted)

Algorithm 2.1 from Epperly et al. (2024): run a sequential rejection sampler
on the `b × b` residual sub-matrix `H`.

- `H` is modified in place (Schur complement updates for accepted pivots).
- `u = diag(H)` is frozen at entry as the acceptance envelope.
- Pivot `j` (1-indexed) is accepted with probability `H[j,j] / u[j]`.
- On acceptance, the Schur complement H[(j+1):b, (j+1):b] is deflated by the
  outer product of the corresponding Cholesky column.
- On rejection, H is left unchanged (the candidate is simply skipped).

Returns the `r × r` lower-triangular Cholesky factor `L` for the accepted
pivots, and the vector of accepted local indices (1-based, within 1:b).
"""
function _rejection_cholesky!(H::Matrix{T}) where {T<:Real}
    b = size(H, 1)
    u = copy(diag(H))          # frozen envelope: u[j] = H[j,j] before any update
    L_scratch = zeros(T, b, b) # working Cholesky columns (b × b)
    accepted  = Int[]

    @inbounds for j in 1:b
        hjj = H[j, j]
        # Skip numerically zero entries (can arise from floating-point negatives)
        hjj <= zero(T) && continue

        # Acceptance test: accept with probability H[j,j] / u[j]
        if rand() * u[j] < hjj
            push!(accepted, j)

            inv_sqrt_hjj = one(T) / sqrt(hjj)

            # Cholesky column for pivot j: L_scratch[j:b, j] = H[j:b, j] / sqrt(H[j,j])
            lj = view(L_scratch, j:b, j)
            lj .= view(H, j:b, j) .* inv_sqrt_hjj

            # Schur complement update: H[j+1:b, j+1:b] -= lj_tail * lj_tail'
            # 5-arg mul! dispatches to BLAS.ger! for Float32/Float64 with no
            # intermediate allocation; falls back to a generic loop otherwise.
            if j < b
                lj_tail = view(L_scratch, j+1:b, j)
                H_sub   = view(H, j+1:b, j+1:b)
                mul!(H_sub, lj_tail, lj_tail', -one(T), one(T))
            end
        end
        # If rejected: H is NOT updated — the diagonal H[j,j] for later pivots
        # is unaffected, preserving the correct marginal distribution.
    end

    r = length(accepted)
    if r == 0
        return zeros(T, 0, 0), Int[]
    end

    # Extract r × r Cholesky factor from the accepted columns
    L = L_scratch[accepted, accepted]
    return L, accepted
end

# ---------------------------------------------------------------------------
# Algorithm 2.2 — AcceleratedRPCholesky core
# ---------------------------------------------------------------------------

"""
    _accelerated_rpcholesky_core!(G, piv, d, get_submatrix!, get_columns!, n, max_rank, rtol, block_size) -> (G, k)

Algorithm 2.2 from Epperly et al. (2024): Accelerated Randomly Pivoted Cholesky.

`G` starts as an `n × block_size` matrix and grows in multiples of `block_size`
as new columns are accepted. `piv` is a `Vector{Int}` that grows via `append!`.

Each outer iteration:
1. Samples `b` candidate indices **with replacement** proportional to `d`.
2. Forms the `b × b` residual sub-matrix `H = A[idx,idx] - G[idx,1:k]*G[idx,1:k]'`.
3. Runs `_rejection_cholesky!(H)` (Algorithm 2.1) to get `r` accepted pivots and
   their `r × r` lower-triangular Cholesky factor `L`.
4. Writes `r` new columns directly into `G[:, k+1:k+r]`, then subtracts the
   current-factor contribution in-place and solves with `L` via `rdiv!`.
5. Updates the residual diagonal `d`.

Returns `(G, k)` — the (possibly grown) factor matrix and total columns written.
"""
function _accelerated_rpcholesky_core!(
    G              :: Matrix{T},
    piv            :: Vector{Int},
    d              :: Vector{T},
    get_submatrix! :: Fsub,
    get_columns!   :: Frows,
    n              :: Int,
    max_rank       :: Int,
    rtol           :: Real,
    block_size     :: Int,
) where {T<:Real, Fsub, Frows}

    trace0 = sum(d)
    trace0 <= zero(T) && return G, 0

    k     = 0
    H_buf = Matrix{T}(undef, block_size, block_size)  # reusable b×b scratch

    while k < max_rank
        # Relative trace stopping criterion
        sum(d) <= rtol * trace0 && break

        b = min(block_size, max_rank - k)

        # Phase 1 — Sample b candidates WITH REPLACEMENT proportional to d
        w = max.(d, zero(T))
        sum(w) <= zero(T) && break
        idx = sample(1:n, Weights(w), b; replace=true)

        # Phase 2 — Form residual sub-matrix H = A[idx,idx] - G[idx,1:k]*G[idx,1:k]'
        H = b == block_size ? H_buf : Matrix{T}(undef, b, b)
        get_submatrix!(H, idx)
        if k > 0
            Gk_idx = G[idx, 1:k]
            mul!(H, Gk_idx, Gk_idx', -one(T), one(T))
        end

        # Phase 3 — Rejection Cholesky on the b×b block
        L, accepted = _rejection_cholesky!(H)
        r = length(accepted)
        r == 0 && continue

        # Cap at remaining budget
        remaining = max_rank - k
        if r > remaining
            r        = remaining
            accepted = accepted[1:r]
            L        = L[1:r, 1:r]
        end

        global_idx = idx[accepted]   # global (1-based) row indices

        # Grow G in multiples of block_size if needed
        if k + r > size(G, 2)
            new_cap = block_size * cld(k + r, block_size)
            new_cap = min(new_cap, n)
            G_new         = Matrix{T}(undef, n, new_cap)
            G_new[:, 1:k] .= view(G, :, 1:k)
            G             = G_new
        end

        # Phase 4 — Write new columns directly into G, update in-place, then solve
        # G[:, k+1:k+r] = (L \ (A[global_idx,:] - G[global_idx,1:k]*G[:,1:k]'))'
        # Equivalent to: G_new * L' = raw   →  rdiv!(G_new, L')
        G_new_cols = view(G, :, k+1:k+r)        # n × r view into G
        get_columns!(G_new_cols', global_idx)    # fill (r × n) = raw columns of G
        if k > 0
            mul!(G_new_cols', G[global_idx, 1:k], view(G, :, 1:k)', -one(T), one(T))
        end
        # Solve G_new_cols * L' = raw  in-place  (no allocation)
        rdiv!(G_new_cols, LowerTriangular(L)')
        append!(piv, global_idx)

        # Phase 5 — Update residual diagonal
        for l in 1:r
            col = view(G, :, k + l)
            @. d = max(d - col * col, zero(T))
        end

        k += r
    end

    return G, k
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    rpcholesky(A; rank=nothing, rtol=0.05, block_size) -> CholeskyPivoted

Compute an Accelerated Randomly Pivoted Cholesky low-rank approximation of the
`n × n` symmetric positive semidefinite matrix `A`.

Returns a `CholeskyPivoted` object `F` where:
- `F.rank` is the number of columns `k` of the computed factor.
- `F.factors[:, 1:F.rank]` is the `n × k` factor `G` satisfying `G * G' ≈ A`.
- `F.p[1:F.rank]` contains the selected pivot indices.

The algorithm (Algorithm 2.2 of Epperly et al., 2024) stops when:
1. `k == rank` (if `rank` is given), **or**
2. `tr(residual) ≤ rtol × tr(A)`.

# Arguments
- `A`           : `n × n` symmetric PSD matrix.
- `rank`        : maximum rank (default: `n`).
- `rtol`        : relative trace tolerance (default: `0.05`).
- `block_size`  : candidates per block; defaults to `clamp(round(Int, sqrt(n)), 4, 256)`.

# Example
```julia
F = rpcholesky(A; rank=20, rtol=0.05)
G = F.factors[:, 1:F.rank]   # n × k low-rank factor
approx = G * G'               # ≈ A
```

# References
Epperly et al. (2024), *Accelerated Randomly Pivoted Cholesky*, arXiv:2410.03969.
"""
function rpcholesky(
    A          :: AbstractMatrix{T};
    rank       :: Union{Int,Nothing} = nothing,
    rtol       :: Real               = 0.05,
    block_size :: Int                = clamp(round(Int, sqrt(LinearAlgebra.checksquare(A))), 4, 256),
) where {T<:Real}
    n        = LinearAlgebra.checksquare(A)
    max_rank = isnothing(rank) ? n : min(rank, n)
    bs       = block_size

    G   = Matrix{T}(undef, n, bs)
    piv = Int[]
    d   = T[A[i,i] for i in 1:n]

    get_submatrix!(H::Matrix{T}, idx::Vector{Int}) = (H .= A[idx, idx])
    get_columns!(R::AbstractMatrix{T}, idx::Vector{Int}) = (R .= A[idx, :])

    G, k = _accelerated_rpcholesky_core!(
        G, piv, d, get_submatrix!, get_columns!, n, max_rank, T(rtol), bs)

    # Embed the n × k factor in an n × n matrix for CholeskyPivoted metadata.
    # NOTE: G is NOT lower-triangular in general (RPCholesky selects random pivots),
    # so use `F.factors[:, 1:F.rank]` to retrieve the actual factor, not `F.L`.
    factors = zeros(T, n, n)
    k > 0 && (factors[:, 1:k] .= view(G, :, 1:k))

    jpvt = vcat(piv, setdiff(1:n, piv))
    return CholeskyPivoted{T, Matrix{T}, Vector{Int}}(
        factors, 'L', jpvt, k, T(rtol), 0)
end

"""
    rpcholesky_kernel(kernel, X; rank=nothing, rtol=0.05, block_size) -> Matrix

Compute an Accelerated Randomly Pivoted Cholesky low-rank factor `G` such that
`G * G' ≈ K`, where `K[i,j] = kernel(X[i,:], X[j,:])`.

Only `O(n·k + b²·iters)` kernel evaluations are performed (vs `O(n²)` for the
full matrix), where `b` is the block size and `iters` is the number of outer
iterations. The acceleration comes from the block rejection sampling step
(Algorithm 2.1 of Epperly et al., 2024), which avoids computing full kernel
columns for rejected candidates.

# Arguments
- `kernel`      : callable `(xᵢ, xⱼ) -> scalar`.
- `X`           : `n × d` data matrix (rows are observations).
- `rank`        : maximum rank (default: `n`).
- `rtol`        : relative trace tolerance (default: `0.05`).
- `block_size`  : candidates per block; defaults to `clamp(round(Int, sqrt(n)), 4, 256)`.

# Returns
An `n × k` matrix `G` such that `G * G' ≈ K`.

# Example
```julia
rbf(x, y) = exp(-sum((x .- y).^2) / 2)
G = rpcholesky_kernel(rbf, X; rank=30, rtol=0.05)
approx = G * G'
```

# References
Epperly et al. (2024), *Accelerated Randomly Pivoted Cholesky*, arXiv:2410.03969.
"""
function rpcholesky_kernel(
    kernel     :: KF,
    X          :: AbstractMatrix{T};
    rank       :: Union{Int,Nothing} = nothing,
    rtol       :: Real               = 0.05,
    block_size :: Int                = clamp(round(Int, sqrt(size(X, 1))), 4, 256),
) where {KF, T<:Real}
    n        = size(X, 1)
    max_rank = isnothing(rank) ? n : min(rank, n)
    bs       = block_size

    # Diagonal: K[i,i] = kernel(xᵢ, xᵢ)
    d = Vector{Float64}(undef, n)
    for i in 1:n
        xi = view(X, i, :)
        d[i] = Float64(kernel(xi, xi))
    end

    # get_submatrix!(H, idx): b×b kernel submatrix (symmetrised)
    function get_submatrix!(H::AbstractMatrix{Float64}, idx::Vector{Int})
        b = length(idx)
        @inbounds for j in 1:b
            xj = view(X, idx[j], :)
            for i in 1:b
                H[i, j] = Float64(kernel(view(X, idx[i], :), xj))
            end
        end
        LinearAlgebra.copytri!(H, 'U')
    end

    # get_columns!(R, idx): fills R (r×n) with kernel rows for idx
    function get_columns!(R::AbstractMatrix{Float64}, idx::Vector{Int})
        r = length(idx)
        @inbounds for j in 1:n
            xj = view(X, j, :)
            for i in 1:r
                R[i, j] = Float64(kernel(view(X, idx[i], :), xj))
            end
        end
    end

    piv = Int[]
    G = Matrix{Float64}(undef, n, bs)
    G, k = _accelerated_rpcholesky_core!(
        G, piv, d, get_submatrix!, get_columns!, n, max_rank,
        Float64(rtol), bs)
    return G[:, 1:k]
end

end # module RPCholesky
