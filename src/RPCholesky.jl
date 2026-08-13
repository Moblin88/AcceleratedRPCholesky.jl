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
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _default_block_size(n)

Heuristic block size: roughly sqrt(n), clipped to [4, 256].
"""
_default_block_size(n::Int) = clamp(round(Int, sqrt(n)), 4, 256)

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
            h_col = view(H, j:b, j)
            lj .= h_col .* inv_sqrt_hjj

            # Schur complement update: deflate remaining sub-block
            # H[j+1:b, j+1:b] -= lj_tail * lj_tail'  (rank-1 update)
            if j < b
                lj_tail = view(L_scratch, j+1:b, j)
                H_sub   = view(H, j+1:b, j+1:b)
                # Generic rank-1 update avoiding a temporary outer-product matrix
                for jj in 1:size(H_sub, 2)
                    ljj = lj_tail[jj]
                    @simd for ii in 1:size(H_sub, 1)
                        H_sub[ii, jj] -= lj_tail[ii] * ljj
                    end
                end
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
# Algorithm 2.2 — AcceleratedRPCholesky core (unified static + dynamic)
# ---------------------------------------------------------------------------

"""
    _accelerated_rpcholesky_core!(G_vec, piv_vec, d, get_submatrix!, get_rows!, ...) -> k

Algorithm 2.2 from Epperly et al. (2024): Accelerated Randomly Pivoted Cholesky.

`G_vec` and `piv_vec` are single-element wrapper vectors holding the current
factor matrix (n × capacity) and pivot vector respectively.  When `dynamic` is
`true`, these are grown with a doubling strategy whenever capacity is exhausted;
when `false`, they are assumed to have exactly `max_rank` columns pre-allocated
and are never resized.

Each outer iteration:
1. Samples `b` candidate indices **with replacement** proportional to `d`.
2. Forms the `b × b` residual sub-matrix `H = A[idx,idx] - G[idx,1:k]*G[idx,1:k]'`.
3. Runs `_rejection_cholesky!(H)` (Algorithm 2.1) to obtain accepted local
   indices and the `r × r` lower-triangular Cholesky factor `L`.
4. For accepted global pivots, computes new factor columns via triangular solve:
   `G[:, k+1:k+r] = (L \\ raw)'`  where `raw = A[global,:]  -  G[global,1:k]*G[:,1:k]'`.
5. Updates the residual diagonal `d`.

Returns the total number of columns `k` written.
"""
function _accelerated_rpcholesky_core!(
    G_vec          :: Vector{Matrix{T}},
    piv_vec        :: Vector{Vector{Int}},
    d              :: Vector{T},
    get_submatrix! :: Fsub,
    get_rows!      :: Frows,
    n              :: Int,
    max_rank       :: Int,
    rtol           :: Real,
    block_size     :: Int,
    dynamic        :: Bool,
) where {T<:Real, Fsub, Frows}

    trace0 = sum(d)
    trace0 <= zero(T) && return 0

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
        G = G_vec[1]
        if k > 0
            Gk_idx = G[idx, 1:k]   # b×k (non-contiguous rows, so a copy is needed)
            mul!(H, Gk_idx, Gk_idx', -one(T), one(T))
        end
        # Symmetrise to guard against floating-point asymmetry
        for jj in 1:b, ii in 1:jj-1
            h = (H[ii, jj] + H[jj, ii]) / 2
            H[ii, jj] = h
            H[jj, ii] = h
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

        # Grow G and piv if needed (dynamic mode only)
        G   = G_vec[1]
        piv = piv_vec[1]
        if dynamic && k + r > size(G, 2)
            new_cap = max(min(size(G, 2) * 2, n), k + r)
            G_new       = Matrix{T}(undef, n, new_cap)
            G_new[:, 1:k] .= view(G, :, 1:k)
            G_vec[1]    = G_new
            G           = G_new

            piv_new     = Vector{Int}(undef, new_cap)
            piv_new[1:k] .= view(piv, 1:k)
            piv_vec[1]  = piv_new
            piv         = piv_new
        end

        # Phase 4 — Compute new factor columns via triangular solve
        # raw = A[global_idx, :] - G[global_idx, 1:k] * G[:, 1:k]'   (r × n)
        raw = Matrix{T}(undef, r, n)
        get_rows!(raw, global_idx)
        if k > 0
            mul!(raw, G[global_idx, 1:k], view(G, :, 1:k)', -one(T), one(T))
        end

        # Solve L * new_cols' = raw  →  new_cols = (L \ raw)'   (n × r)
        new_cols_T = LowerTriangular(L) \ raw   # r × n
        G[:, k+1:k+r]  .= new_cols_T'
        piv[k+1:k+r]    .= global_idx

        # Phase 5 — Update residual diagonal
        for l in 1:r
            col = view(G, :, k + l)
            @. d = max(d - col * col, zero(T))
        end

        k += r
    end

    return k
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    rpcholesky(A; rank=nothing, rtol=0.05, block_size=nothing) -> CholeskyPivoted

Compute an Accelerated Randomly Pivoted Cholesky low-rank approximation of the
`n × n` symmetric positive semidefinite matrix `A`.

Returns a `CholeskyPivoted` object `F` where:
- `F.rank` is the number of columns `k` of the computed factor.
- `F.factors[:, 1:F.rank]` is the `n × k` factor `L` satisfying `L * L' ≈ A`.
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
L = F.factors[:, 1:F.rank]   # n × k low-rank factor
approx = L * L'               # ≈ A
```

# References
Epperly et al. (2024), *Accelerated Randomly Pivoted Cholesky*, arXiv:2410.03969.
"""
function rpcholesky(
    A          :: AbstractMatrix{T};
    rank       :: Union{Int,Nothing} = nothing,
    rtol       :: Real               = 0.05,
    block_size :: Union{Int,Nothing} = nothing,
) where {T<:Real}
    n        = LinearAlgebra.checksquare(A)
    max_rank = isnothing(rank) ? n : min(rank, n)
    bs       = isnothing(block_size) ? _default_block_size(n) : block_size

    G   = Matrix{T}(undef, n, max_rank)
    piv = Vector{Int}(undef, max_rank)
    d   = T[A[i,i] for i in 1:n]

    # get_submatrix!(H, idx): H[i,j] = A[idx[i], idx[j]]
    function get_submatrix!(H::Matrix{T}, idx::Vector{Int})
        b = length(idx)
        @inbounds for j in 1:b, i in 1:b
            H[i, j] = A[idx[i], idx[j]]
        end
    end

    # get_rows!(R, idx): R[i, :] = A[idx[i], :]
    function get_rows!(R::Matrix{T}, idx::Vector{Int})
        r = length(idx)
        @inbounds for j in 1:n, i in 1:r
            R[i, j] = A[idx[i], j]
        end
    end

    G_vec   = [G]
    piv_vec = [piv]
    k = _accelerated_rpcholesky_core!(
        G_vec, piv_vec, d, get_submatrix!, get_rows!, n, max_rank, T(rtol), bs, false)
    G   = G_vec[1]
    piv = piv_vec[1]

    # Embed the n × k factor in an n × n matrix for CholeskyPivoted metadata.
    # NOTE: G is NOT lower-triangular in general (RPCholesky selects random pivots),
    # so use `F.factors[:, 1:F.rank]` to retrieve the actual factor, not `F.L`.
    factors = zeros(T, n, n)
    k > 0 && (factors[:, 1:k] .= view(G, :, 1:k))

    jpvt = vcat(piv[1:k], setdiff(1:n, piv[1:k]))
    return CholeskyPivoted{T, Matrix{T}, Vector{Int}}(
        factors, 'L', jpvt, k, T(rtol), 0)
end

"""
    rpcholesky_kernel(kernel, X; rank=nothing, rtol=0.05, block_size=nothing) -> Matrix

Compute an Accelerated Randomly Pivoted Cholesky low-rank factor `L` such that
`L * L' ≈ K`, where `K[i,j] = kernel(X[i,:], X[j,:])`.

Only `O(n·k + b²·iters)` kernel evaluations are performed (vs `O(n²)` for the
full matrix), where `b` is the block size and `iters` is the number of outer
iterations. The acceleration comes from the block rejection sampling step
(Algorithm 2.1 of Epperly et al., 2024), which avoids computing full kernel
columns for rejected candidates.

# Arguments
- `kernel`      : callable `(xᵢ, xⱼ) -> scalar`.
- `X`           : `n × d` data matrix (rows are observations).
- `rank`        : maximum rank (default: `n`, dynamically allocated).
- `rtol`        : relative trace tolerance (default: `0.05`).
- `block_size`  : candidates per block; defaults to `clamp(round(Int, sqrt(n)), 4, 256)`.

# Returns
An `n × k` matrix `L` such that `L * L' ≈ K`.

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
    kernel     :: KF,
    X          :: AbstractMatrix{T};
    rank       :: Union{Int,Nothing} = nothing,
    rtol       :: Real               = 0.05,
    block_size :: Union{Int,Nothing} = nothing,
) where {KF, T<:Real}
    n        = size(X, 1)
    max_rank = isnothing(rank) ? n : min(rank, n)
    bs       = isnothing(block_size) ? _default_block_size(n) : block_size

    # Diagonal: K[i,i] = kernel(xᵢ, xᵢ)
    d = Vector{Float64}(undef, n)
    for i in 1:n
        xi = view(X, i, :)
        d[i] = Float64(kernel(xi, xi))
    end

    # get_submatrix!(H, idx): b×b kernel submatrix
    function get_submatrix!(H::Matrix{Float64}, idx::Vector{Int})
        b = length(idx)
        @inbounds for j in 1:b
            xj = view(X, idx[j], :)
            for i in 1:b
                H[i, j] = Float64(kernel(view(X, idx[i], :), xj))
            end
        end
    end

    # get_rows!(R, idx): r×n kernel rows
    function get_rows!(R::Matrix{Float64}, idx::Vector{Int})
        r = length(idx)
        @inbounds for j in 1:n
            xj = view(X, j, :)
            for i in 1:r
                R[i, j] = Float64(kernel(view(X, idx[i], :), xj))
            end
        end
    end

    if isnothing(rank)
        # Dynamic allocation: start small and double as needed
        init_cols = min(_default_block_size(n) * 2, n)
        G_vec   = [Matrix{Float64}(undef, n, init_cols)]
        piv_vec = [Vector{Int}(undef, init_cols)]
        k = _accelerated_rpcholesky_core!(
            G_vec, piv_vec, d, get_submatrix!, get_rows!, n, max_rank,
            Float64(rtol), bs, true)
        return G_vec[1][:, 1:k]
    else
        G   = Matrix{Float64}(undef, n, max_rank)
        piv = Vector{Int}(undef, max_rank)
        G_vec   = [G]
        piv_vec = [piv]
        k = _accelerated_rpcholesky_core!(
            G_vec, piv_vec, d, get_submatrix!, get_rows!, n, max_rank,
            Float64(rtol), bs, false)
        return G_vec[1][:, 1:k]
    end
end

end # module RPCholesky
