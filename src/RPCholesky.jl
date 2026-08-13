"""
    RPCholesky

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
module RPCholesky

using LinearAlgebra
using StatsBase: sample, Weights

export rpcholesky

# Algorithm 2.1 from Epperly et al. (2024).
# Runs a sequential rejection sampler on the b×b residual sub-matrix H.
# u = diag(H) is frozen at entry as the acceptance envelope; pivot j is
# accepted with probability H[j,j] / u[j]. On acceptance, H is deflated by
# the Schur complement; on rejection, H is left unchanged.
# Returns the r×r lower-triangular Cholesky factor L and the accepted local indices.
function _rejection_cholesky!(H::Matrix{Float64})
    b         = size(H, 1)
    u         = copy(diag(H))
    L_scratch = zeros(Float64, b, b)
    accepted  = Int[]

    @inbounds for j in 1:b
        hjj = H[j, j]
        hjj <= 0.0 && continue
        if rand() * u[j] < hjj
            push!(accepted, j)
            lj = view(L_scratch, j:b, j)
            lj .= view(H, j:b, j) ./ sqrt(hjj)
            if j < b
                lj_tail = view(L_scratch, j+1:b, j)
                mul!(view(H, j+1:b, j+1:b), lj_tail, lj_tail', -1.0, 1.0)
            end
        end
    end

    isempty(accepted) && return zeros(Float64, 0, 0), accepted
    return L_scratch[accepted, accepted], accepted
end

"""
    rpcholesky(kernel, X; rank=n, rtol=0.05, block_size) -> Matrix

Compute an Accelerated Randomly Pivoted Cholesky low-rank factor `G` such that
`G * G' ≈ K`, where `K[i,j] = kernel(X[i,:], X[j,:])`.

Only `O(n·k + b²·iters)` kernel evaluations are performed (vs `O(n²)` for the
full matrix), where `b` is the block size and `iters` is the number of outer
iterations. The acceleration comes from the block rejection sampling step
(Algorithm 2.1 of Epperly et al., 2024), which avoids computing full kernel
columns for rejected candidates.

The algorithm stops when:
1. `k == rank` (if `rank` is given), **or**
2. `tr(residual) ≤ rtol × tr(K)`.

# Arguments
- `kernel`      : callable `(xᵢ, xⱼ) -> scalar`.
- `X`           : `n × d` data matrix (rows are observations).
- `rank`        : maximum rank (default: `n`).
- `rtol`        : relative trace tolerance (default: `0.05`).
- `block_size`  : candidates per block; defaults to `clamp(round(Int, sqrt(n)), 4, 256)`.

# Returns
An `n × k` matrix `G` such that `G * G'` approximates the kernel matrix.

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
    kernel     :: KF,
    X          :: AbstractMatrix{T};
    rank       :: Int  = size(X, 1),
    rtol       :: Real = 0.05,
    block_size :: Int  = clamp(round(Int, sqrt(size(X, 1))), 4, 256),
) where {KF, T<:Real}
    n        = size(X, 1)
    max_rank = min(rank, n)

    # Residual diagonal: d[i] = K[i,i] = kernel(xᵢ, xᵢ)
    d = Float64[kernel(view(X, i, :), view(X, i, :)) for i in 1:n]

    trace0 = sum(d)
    trace0 <= 0.0 && return Matrix{Float64}(undef, n, 0)

    G     = Matrix{Float64}(undef, n, block_size)
    k     = 0
    H_buf = Matrix{Float64}(undef, block_size, block_size)

    while k < max_rank
        sum(d) <= rtol * trace0 && break

        b = min(block_size, max_rank - k)

        # Phase 1 — sample b candidates proportional to residual diagonal
        w = max.(d, 0.0)
        sum(w) <= 0.0 && break
        idx = sample(1:n, Weights(w), b; replace=true)

        # Phase 2 — form b×b residual submatrix H = K[idx,idx] - G[idx,1:k]*G[idx,1:k]'
        # Fill lower triangle only (kernel is symmetric), then mirror
        H = b == block_size ? H_buf : Matrix{Float64}(undef, b, b)
        @inbounds for j in 1:b
            xj = view(X, idx[j], :)
            for i in j:b
                H[i, j] = kernel(view(X, idx[i], :), xj)
            end
        end
        copytri!(H, 'L')
        if k > 0
            Gk = G[idx, 1:k]
            mul!(H, Gk, Gk', -1.0, 1.0)
        end

        # Phase 3 — rejection Cholesky on the b×b block (Algorithm 2.1)
        L, accepted = _rejection_cholesky!(H)
        r = length(accepted)
        r == 0 && continue

        # Cap at remaining budget
        if r > max_rank - k
            r        = max_rank - k
            accepted = accepted[1:r]
            L        = L[1:r, 1:r]
        end

        global_idx = idx[accepted]

        # Grow G in multiples of block_size if needed
        if k + r > size(G, 2)
            new_cap       = min(block_size * cld(k + r, block_size), n)
            G_new         = Matrix{Float64}(undef, n, new_cap)
            G_new[:, 1:k] .= view(G, :, 1:k)
            G             = G_new
        end

        # Phase 4 — fill new columns, subtract prior contribution, solve in-place
        # G[:, k+1:k+r] * L' = K[:, global_idx] - G[:,1:k] * G[global_idx,1:k]'
        G_new_cols = view(G, :, k+1:k+r)
        @inbounds for i in eachindex(global_idx)
            xi = view(X, global_idx[i], :)
            for j in 1:n
                G_new_cols[j, i] = kernel(view(X, j, :), xi)
            end
        end
        if k > 0
            mul!(G_new_cols, view(G, :, 1:k), G[global_idx, 1:k]', -1.0, 1.0)
        end
        rdiv!(G_new_cols, LowerTriangular(L)')

        # Phase 5 — update residual diagonal
        for l in 1:r
            col = view(G, :, k + l)
            @. d = max(d - col * col, 0.0)
        end

        k += r
    end

    return G[:, 1:k]
end

end # module RPCholesky

