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
    L_scratch = similar(H, b, b)
    accepted  = Int[]
    sizehint!(accepted, min(b, max_accepted))

    @inbounds for j in 1:b
        length(accepted) == max_accepted && break
        hjj = H[j, j]
        hjj <= abs_tol && continue
        if rand() * u[j] < hjj
            push!(accepted, j)
            lj = view(L_scratch, j:b, j)
            lj .= view(H, j:b, j) ./ sqrt(hjj)
            if j < b
                lj_tail = view(L_scratch, j+1:b, j)
                mul!(view(H, j+1:b, j+1:b), lj_tail, lj_tail', -one(eltype(H)), one(eltype(H)))
            end
        end
    end
    return LowerTriangular(L_scratch[accepted, accepted]), accepted
end

"""
    rpcholesky(kernel, data; rank=n, rtol=0.05, atol=1e-8, block_size) -> Matrix

Compute an Accelerated Randomly Pivoted Cholesky low-rank factor `G` such that
`G * G' ≈ K`, where `K[i,j] = kernel(data[i,:], data[j,:])`. `data` may be
any one-based, row-indexable collection that supports `size(data, 1)` and
`view(data, i, :)`; this includes ordinary matrices and `DataFrame`s.

Only `O(n·k + b²·iters)` kernel evaluations are performed (vs `O(n²)` for the
full matrix), where `b` is the block size and `iters` is the number of outer
iterations. The acceleration comes from the block rejection sampling step
(Algorithm 2.1 of Epperly et al., 2024), which avoids computing full kernel
columns for rejected candidates.

The algorithm stops when:
1. `k == rank` (if `rank` is given), **or**
2. `tr(residual) ≤ rtol × tr(K)`.

# Arguments
- `kernel`      : callable `(xᵢ, xⱼ) -> scalar`. The scalar type is preserved
                  in the returned factor, so kernels returning `Float32`
                  produce a `Float32` factor.
- `data`        : `n × d` row-indexable data collection (for example, a
                  matrix or `DataFrame`). Rows are passed to `kernel` as
                  `view(data, i, :)`.
- `rank`        : maximum rank (default: `n`).
- `rtol`        : relative trace tolerance (default: `0.05`).
- `atol`        : absolute residual trace cutoff (default: `1e-8`).
- `block_size`  : candidates per block; defaults to `clamp(round(Int, sqrt(n)), 4, 256)`.

# Returns
An `n × k` matrix `G` whose element type matches the kernel diagonal values,
such that `G * G'` approximates the kernel matrix. The number of columns is
at most `rank` and may be smaller when either tolerance is reached.

# Example
```julia
rbf(x, y) = exp(-sum((x .- y).^2) / 2)
G = rpcholesky(rbf, X; rank=30, rtol=0.05)
approx = G * G'
```

# References
Epperly et al. (2024), *Accelerated Randomly Pivoted Cholesky*, arXiv:2410.03969.
"""
function rpcholesky(
    kernel,
    data;
    rank = size(data, 1),
    rtol = 0.05,
    atol = 1e-8,
    block_size = clamp(round(Int, sqrt(size(data, 1))), 4, 256),
)
    Base.require_one_based_indexing(data)
    n = size(data, 1)
    max_rank = min(rank, n)

    # Residual diagonal: d[i] = K[i,i] = kernel(xᵢ, xᵢ)
    d = [kernel(view(data, i, :), view(data, i, :)) for i in 1:n]

    trace_mass = trace0 = sum(d)

    G = similar(d, n, min(block_size * 2, max_rank))
    k = 0
    H_buf = similar(d, block_size, block_size)

    while true
        k == max_rank && break # reached maximum rank
        trace_mass <= rtol * trace0 && break # explained enough trace mass
        trace_mass <= atol && break # rank deficiency (numerical zero)

        b = min(block_size, n - k)

        # Phase 1 — sample b candidates proportional to residual diagonal
        idx = sample(1:n, d, b)

        # Phase 2 — form b×b residual submatrix H = K[idx,idx] - G[idx,1:k]*G[idx,1:k]'
        # Fill lower triangle only (kernel is symmetric), then mirror
        H = view(H_buf, 1:b, 1:b)
        @inbounds for j in 1:b
            xj = view(data, idx[j], :)
            for i in j:b
                H[i, j] = kernel(view(data, idx[i], :), xj)
            end
        end
        LinearAlgebra.copytri!(H, 'L')

        Gk = G[idx, 1:k] # not a view since we need a contiguous block for BLAS
        mul!(H, Gk, Gk', -true, true)

        # Phase 3 — rejection Cholesky on the b×b block (Algorithm 2.1)
        L, accepted = _block_pivot_cholesky!(H, max_rank - k, atol)
        r = length(accepted)
        r == 0 && continue

        global_idx = idx[accepted]

        # Grow G if nesessary
        # try to estimate how long until we reach r_tol * trace0
        if k + r > size(G, 2)
            new_cap       = clamp(ceil(Int, k * trace0 * (1-rtol) / (trace0 - trace_mass)), k+r, max_rank)
            G_new         = similar(G, n, new_cap)
            G_new[:, 1:k] .= G[:, 1:k]
            G             = G_new
        end

        # Phase 4 — fill new columns, subtract prior contribution, solve in-place
        # G[:, k+1:k+r] * L' = K[:, global_idx] - G[:,1:k] * G[global_idx,1:k]'
        G_new_cols = view(G, :, k+1:k+r)
        @inbounds for i in eachindex(global_idx)
            xi = view(data, global_idx[i], :)
            for j in 1:n
                G_new_cols[j, i] = kernel(view(data, j, :), xi)
            end
        end
        mul!(G_new_cols, view(G, :, 1:k), G[global_idx, 1:k]', -one(eltype(G)), one(eltype(G)))
        rdiv!(G_new_cols, L')

        # Phase 5 — update residual diagonal and residual trace mass
        trace_mass = 0.0
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

    return G[:, 1:k]
end

end # module AcceleratedRPCholesky
