"""
    AcceleratedRPCholesky

Julia package implementing the Accelerated Randomly Pivoted Cholesky (RPCholesky)
algorithm for low-rank approximation of kernel matrices.

The low-rank factor `G` (n × k) satisfies `G * G' ≈ K`, where `K` is the kernel
matrix induced by a user-supplied kernel function.

# References
Chen, Y., Epperly, E. N., Tropp, J. A., & Webber, R. J. (2024).
*Randomly pivoted Cholesky: Practical approximation of a kernel matrix with
few entry evaluations.*
[arXiv:2207.06503](https://arxiv.org/abs/2207.06503)

Epperly, E. N., Frangella, Z., Tropp, J. A., Webber, R. J., & Zangrando, M. (2024).
*Accelerated Randomly Pivoted Cholesky.*
[arXiv:2410.03969](https://arxiv.org/abs/2410.03969)
"""
module AcceleratedRPCholesky

using LinearAlgebra
using Random

export rpcholesky

function sample!(selected, cum_weights, weights, rng)
    cumsum!(cum_weights, weights)
    total_weight = last(cum_weights)
    for i in eachindex(selected)
        r = rand(rng) * total_weight
        idx = searchsortedfirst(cum_weights, r)
        selected[i] = idx
    end
    return nothing
end

# Algorithm 2.1 from Epperly et al. (2024).
# Runs a sequential rejection sampler on the b×b residual sub-matrix H.
# u = diag(H) is frozen at entry as the acceptance envelope; pivot j is
# accepted with probability H[j,j] / u[j]. On acceptance, H is deflated by
# the Schur complement; on rejection, H is left unchanged.
# Returns the r×r lower-triangular Cholesky factor L and accepted count r.
function _block_pivot_cholesky!(H, idx, max_accepted, abs_tol, u, rng = Random.default_rng())
    Base.require_one_based_indexing(H)
    b = LinearAlgebra.checksquare(H)
    @inbounds for i in 1:b
        u[i] = H[i, i]
    end
    r = 0

    @inbounds for j in 1:b
        r == max_accepted && break
        hjj = H[j, j]
        if hjj > abs_tol && rand(rng) * u[j] < hjj
            r += 1
            idx[r] = idx[j]
            sqrt_hjj = sqrt(hjj)
            H[r,r] = sqrt_hjj
            inv_sqrt_hjj = inv(sqrt_hjj)
            lj_tail = view(H, j+1:b, r)
            mul!(lj_tail, view(H, j+1:b, j), inv_sqrt_hjj)
            mul!(view(H, j+1:b, j+1:b), lj_tail, lj_tail', -one(eltype(H)), one(eltype(H))) #should use syrk! and an (unneeded) copy
            if r < j
                for y in 1:r-1
                    H[r, y] = H[j, y]
                end
            end
        end
    end

    return LowerTriangular(view(H, 1:r, 1:r)), r
end

"""
    rpcholesky(kernel, data, [Val(true)];
               rank=min(n, 50), rtol=0.05, atol=1e-8, block_size, rng=Random.default_rng())

Compute an Accelerated Randomly Pivoted Cholesky low-rank factor `G` such that
`G * G' ≈ K`, where `K[i,j] = kernel(data[i,:], data[j,:])`. `data` may be
any one-based, row-indexable collection that supports `size(data, 1)` and
`view(data, i, :)`; this includes ordinary matrices and `DataFrame`s.

Only `O(n·k + b²·iters)` kernel evaluations are performed (vs `O(n²)` for the
full matrix), where `b` is the block size and `iters` is the number of outer
iterations. The acceleration comes from the block rejection sampling step
(Algorithm 2.1 of Epperly et al., 2024), which avoids computing full kernel
columns for rejected candidates.

The algorithm stops when either:
1. `k == rank` (requested rank is reached), **or**
2. `tr(residual) ≤ rtol * tr(K)` (the relative residual trace target is reached), **or**
3. `tr(residual) ≤ atol` (the absolute residual trace cutoff is reached).

Whichever happens first.

# Arguments
- `kernel`      : callable `(xᵢ, xⱼ) -> scalar`. The scalar type is preserved
                  in the returned factor, so kernels returning `Float32`
                  produce a `Float32` factor.
- `data`        : `n × d` row-indexable data collection (for example, a
                  matrix or `DataFrame`). Rows are passed to `kernel` as
                  `view(data, i, :)`.
- `rank`        : maximum rank to compute (default: `min(n, 50)`).
- `rtol`        : relative residual trace cutoff (default: `0.05`).
- `atol`        : absolute residual trace cutoff (default: `1e-8`).
- `block_size`  : candidates per block; defaults to `clamp(round(Int, sqrt(n)), 4, 24)`.
- `rng`         : random number generator used for both candidate sampling and
                  rejection sampling (default: `Random.default_rng()`).

# Memory note
`rpcholesky` allocates storage for up to `rank` columns of `G` up front
(`n × rank`), then truncates to the accepted rank `k` on return. Choose large
`rank` values with care: allocation cost is paid even if stopping triggers
early from `rtol` or `atol`.

# Returns
- `rpcholesky(kernel, data; ...)` returns `G`, an `n × k` matrix whose element
  type matches the kernel diagonal values.
- `rpcholesky(kernel, data, Val(true); ...)` returns `(G, pivots)`, where
  `pivots` is an integer vector containing the global indices of the accepted
  pivot rows and `k = length(pivots)`.

# Example
```julia
rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2)
G = rpcholesky(rbf, X)
G = rpcholesky(rbf, X; rank=30, rtol=0.01)
G, pivots = rpcholesky(rbf, X, Val(true); rank=30, rtol=0.01)
```

# References
Epperly et al. (2024), *Accelerated Randomly Pivoted Cholesky*, arXiv:2410.03969.
"""
function rpcholesky(
    kernel,
    data,
    ::Val{P} = Val(false);
    rank = min(size(data, 1), 50),
    rtol = 0.05,
    atol = 1e-8,
    block_size = clamp(round(Int, sqrt(size(data, 1))), 4, 24),
    rng = Random.default_rng()
) where {P}
    Base.require_one_based_indexing(data)
    n = size(data, 1)
    0 <= rtol < 1 || throw(ArgumentError("rtol must be in [0, 1)"))
    0 <= rank <= n || throw(ArgumentError("rank must be in [0, n]"))
    @inbounds trace_mass = kernel(view(data, 1, :), view(data, 1, :))
    d = Vector{typeof(trace_mass)}(undef, n)
    @inbounds d[1] = trace_mass
    @inbounds for i in 2:n
        val = kernel(view(data, i, :), view(data, i, :))
        d[i] = val
        trace_mass += val
    end
    trace0 = trace_mass
    G = similar(d, n, rank)
    k = 0
    H = similar(d, block_size, block_size)
    Gk = similar(d, block_size, rank)
    idx = Vector{Int}(undef, block_size)
    u = Vector{eltype(H)}(undef, block_size)
    cum_weights = similar(d)
    pivots = P ? Int[] : nothing
    # Keep this in the pivot-returning specialization only; an unconditional
    # sizehint! is effectful and not eliminated in the Val(false) path.
    P && sizehint!(pivots, rank)

    while k < rank
        trace_mass <= rtol * trace0 && break # relative tolerance satisfied
        trace_mass <= atol && break # rank deficiency (numerical zero)

        # Phase 1 — sample block_size candidates proportional to residual diagonal
        sample!(idx, cum_weights, d, rng)

        # Phase 2 — form block_size×block_size residual submatrix H = K[idx,idx] - G[idx,1:k]*G[idx,1:k]'
        # Fill lower triangle only
        @inbounds for j in 1:block_size
            xj = view(data, idx[j], :)
            for i in j:block_size
                H[i, j] = kernel(view(data, idx[i], :), xj)
            end
        end
        # copy up, even though we don't need it, so that we can use BLAS for the Schur complement update
        LinearAlgebra.copytri!(H,'L')

        Gk[:, 1:k] .= G[idx, 1:k] # not a view since we need a contiguous block for BLAS
        Gk_view = view(Gk, :, 1:k)
        # will use syrk! and copy if H is symmetric (but not a Symmetric type).
        # Strictly speaking, this is more than we need since we only need the lower triangle
        # However, by not using BLAS directly we remain usable on non-BLAS numeric types
        mul!(H, Gk_view, Gk_view', -one(eltype(H)), one(eltype(H)))
        # Phase 3 — rejection Cholesky on the b×b block (Algorithm 2.1)
        L, r = _block_pivot_cholesky!(H, idx, rank - k, atol, u, rng)
        r == 0 && continue

        global_idx_view = view(idx, 1:r)
        P && append!(pivots, global_idx_view)

        # Phase 4 — fill new columns, subtract prior contribution, solve in-place
        # G[:, k+1:k+r] * L' = K[:, global_idx] - G[:,1:k] * G[global_idx,1:k]'
        G_new_cols = view(G, :, k+1:k+r)
        @inbounds for i in 1:r
            xi = view(data, global_idx_view[i], :)
            for j in 1:n
                G_new_cols[j, i] = kernel(view(data, j, :), xi)
            end
        end
        Gk[1:r,1:k] .= G[global_idx_view, 1:k] # not a view since we need a contiguous block for BLAS
        mul!(G_new_cols, view(G, :, 1:k), view(Gk, 1:r, 1:k)', -one(eltype(G)), one(eltype(G))) # views are all still strided arrays and BLASable
        rdiv!(G_new_cols, L')

        # Phase 5 — update residual diagonal and residual trace mass
        trace_mass = zero(trace_mass)
        @inbounds for i in 1:n
            dᵢ = d[i]
            for l in 1:r
                dᵢ -= abs2(G_new_cols[i, l])
            end
            dᵢ = max(dᵢ, 0.0)
            d[i] = dᵢ
            trace_mass += dᵢ
        end

        k += r
    end
    if k < rank
        G = G[:, 1:k]
    end
    if P
        return G, pivots
    else
        return G
    end
end

end # module AcceleratedRPCholesky
