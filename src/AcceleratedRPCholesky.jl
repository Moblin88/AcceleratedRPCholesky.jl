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

function sample(items, weights, count)
    selected = similar(items, count)
    cum_weights = cumsum(weights)
    total_weight = last(cum_weights)
    for i in eachindex(selected)
        r = rand() * total_weight
        idx = searchsortedfirst(cum_weights, r)
        selected[i] = @inbounds items[idx]
    end
    return selected
end

# Algorithm 2.1 from Epperly et al. (2024).
# Runs a sequential rejection sampler on the b×b residual sub-matrix H.
# u = diag(H) is frozen at entry as the acceptance envelope; pivot j is
# accepted with probability H[j,j] / u[j]. On acceptance, H is deflated by
# the Schur complement; on rejection, H is left unchanged.
# Returns the r×r lower-triangular Cholesky factor L and the accepted local indices.
function _block_pivot_cholesky!(H, max_accepted, abs_tol)
    Base.require_one_based_indexing(H)
    b = LinearAlgebra.checksquare(H)
    u = diag(H)
    accepted  = Int[]
    sizehint!(accepted, min(b, max_accepted))

    @inbounds for j in 1:b
        length(accepted) == max_accepted && break
        hjj = H[j, j]
        hjj <= abs_tol && continue
        if rand() * u[j] < hjj
            push!(accepted, j)
            H[j:b,j] ./= sqrt(hjj)
            lj_tail = view(H, j+1:b, j)
            mul!(view(H, j+1:b, j+1:b), lj_tail, lj_tail', -one(eltype(H)), one(eltype(H)))
        end
    end
    return LowerTriangular(H[accepted, accepted]), accepted
end

"""
    rpcholesky(kernel, data; rank=min(n, 50), rtol=0.05, atol=1e-8, block_size) -> (Matrix, Vector)

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

# Returns
A tuple `(G, pivots)` where:
- `G` is an `n × k` matrix whose element type matches the kernel diagonal values,
  where `k = length(pivots)` is the accepted rank.
- `pivots` is an integer vector containing the global indices of the
  accepted pivot rows.

# Example
```julia
rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2)
G, pivots = rpcholesky(rbf, X)
G, pivots = rpcholesky(rbf, X; rank=30, rtol=0.01)
```

# References
Epperly et al. (2024), *Accelerated Randomly Pivoted Cholesky*, arXiv:2410.03969.
"""
function rpcholesky(
    kernel,
    data;
    rank = min(size(data, 1), 50),
    rtol = 0.05,
    atol = 1e-8,
    block_size = clamp(round(Int, sqrt(size(data, 1))), 4, 24)
)
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
    pivots = Int[]
    sizehint!(pivots, rank)

    while k < rank
        trace_mass <= rtol * trace0 && break # relative tolerance satisfied
        trace_mass <= atol && break # rank deficiency (numerical zero)

        # Phase 1 — sample block_size candidates proportional to residual diagonal
        idx = sample(1:n, d, block_size)

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
        L, accepted = _block_pivot_cholesky!(H, rank - k, atol)
        r = length(accepted)
        r == 0 && continue

        global_idx = idx[accepted]
        append!(pivots, global_idx)

        # Phase 4 — fill new columns, subtract prior contribution, solve in-place
        # G[:, k+1:k+r] * L' = K[:, global_idx] - G[:,1:k] * G[global_idx,1:k]'
        G_new_cols = view(G, :, k+1:k+r)
        @inbounds for i in eachindex(global_idx)
            xi = view(data, global_idx[i], :)
            for j in 1:n
                G_new_cols[j, i] = kernel(view(data, j, :), xi)
            end
        end
        Gk[1:r,1:k] .= G[global_idx, 1:k] # not a view since we need a contiguous block for BLAS
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
    return G, pivots
end

end # module AcceleratedRPCholesky
