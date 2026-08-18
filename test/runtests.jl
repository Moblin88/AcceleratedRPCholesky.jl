using Test
using LinearAlgebra
using Random
using DataFrames
using AcceleratedRPCholesky

mutable struct SequenceRNG <: AbstractRNG
    values::Vector{Float64}
    index::Int
end

function Random.rand(rng::SequenceRNG)
    value = rng.values[rng.index]
    rng.index += 1
    return value
end

Random.seed!(42)

@testset "AcceleratedRPCholesky.jl" begin

    @testset "_block_pivot_cholesky! in-place factorization" begin
        A = [
            1.0 0.2 0.3
            0.4 1.1 0.1
            0.2 0.5 0.9
            0.3 0.1 0.7
            0.6 0.4 0.2
            0.1 0.8 0.3
        ]
        H0 = A * A' + I
        H = copy(H0)
        b = size(H, 1)
        idx = collect(1:b)
        u = Vector{eltype(H)}(undef, b)

        Random.seed!(42)
        L, r = AcceleratedRPCholesky._block_pivot_cholesky!(
            H,
            idx,
            size(H, 1),
            1e-12,
            u,
        )

        @test r > 0
        @test H != H0
        @test Matrix(L * L') ≈ H0[view(idx, 1:r), view(idx, 1:r)]
    end

    @testset "_block_pivot_cholesky! skips candidates" begin
        H0 = [
            2.0 0.4 0.3
            0.4 1.5 0.2
            0.3 0.2 1.2
        ]
        H = copy(H0)
        idx = collect(1:size(H, 1))
        u = Vector{eltype(H)}(undef, size(H, 1))
        rng = SequenceRNG([0.0, 1.0, 0.0], 1) # Accept 1, reject 2, accept 3.

        L, r = AcceleratedRPCholesky._block_pivot_cholesky!(
            H,
            idx,
            2,
            1e-12,
            u,
            rng,
        )

        selected = view(idx, 1:r)
        @test selected == [1, 3]
        @test Matrix(L) ≈ cholesky(H0[selected, selected]).L
    end

    @testset "rpcholesky - kernel interface" begin
        n = 15
        d = 3
        X = randn(n, d)

        rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2.0)

        K = [rbf(X[i,:], X[j,:]) for i in 1:n, j in 1:n]

        @testset "default returns a matrix" begin
            G = rpcholesky(rbf, X; rank=5)
            @test G isa Matrix
            @test size(G, 1) == n
            @test size(G, 2) <= 5
        end

        @testset "Val(true) returns pivots" begin
            G, pivots = rpcholesky(rbf, X, Val(true); rank=5)
            @test G isa Matrix
            @test pivots isa Vector{Int}
            @test length(pivots) == size(G, 2)
        end

        @testset "supports DataFrames" begin
            data = DataFrame(x=X[:, 1], y=X[:, 2], z=X[:, 3])
            dataframe_rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2.0)
            G = rpcholesky(dataframe_rbf, data; rank=5)
            @test G isa Matrix{Float64}
            @test size(G, 1) == n
            @test size(G, 2) <= 5
        end

        @testset "preserves Float32 kernel type" begin
            X32 = Float32.(X)
            rbf32(x, y) = exp(Float32(-sum((x .- y).^2) / 2))
            G = rpcholesky(rbf32, X32; rank=5)
            @test G isa Matrix{Float32}
            @test size(G, 1) == n
            @test size(G, 2) <= 5
            @test eltype(G * G') == Float32
        end

        @testset "approximation quality" begin
            G = rpcholesky(rbf, X; rank=n, rtol=1e-6)
            approx = G * G'
            err = norm(K - approx, 1) / norm(K, 1)
            @test err < 0.5
        end

        @testset "keyword rank defaults to min(n, 50)" begin
            X_default = randn(51, d)
            G = rpcholesky(rbf, X_default)
            @test size(G, 1) == size(X_default, 1)
            @test size(G, 2) <= 50
        end

        @testset "block_size parameter accepted" begin
            G = rpcholesky(rbf, X; rank=3, block_size=2)
            @test size(G, 2) <= 3
        end

        @testset "rng keyword provides reproducibility" begin
            rng1 = MersenneTwister(123)
            rng2 = MersenneTwister(123)
            G1 = rpcholesky(rbf, X; rank=8, block_size=4, rng=rng1)
            G2 = rpcholesky(rbf, X; rank=8, block_size=4, rng=rng2)
            @test G1 == G2
        end

        @testset "rtol stopping" begin
            Random.seed!(42)
            G_tight = rpcholesky(rbf, X; rank=n, rtol=0.01)
            Random.seed!(42)
            G_loose = rpcholesky(rbf, X; rank=n, rtol=0.9)
            @test size(G_loose, 2) <= size(G_tight, 2)
        end

        @testset "residual trace tolerance" begin
            Random.seed!(42)
            rtol = 0.2
            G = rpcholesky(rbf, X; rank=n, rtol=rtol, block_size=4)
            residual = K - G * G'
            @test tr(Symmetric(residual)) <= rtol * tr(Symmetric(K)) + 1e-8
        end

        @testset "zero kernel returns empty factor" begin
            zero_kernel(x, y) = 0.0
            G = rpcholesky(zero_kernel, X; rank=5)
            @test G isa Matrix
            @test size(G, 2) == 0
        end

        @testset "rank greater than block size" begin
            Random.seed!(42)
            G = rpcholesky(rbf, X; rank=8, block_size=1)
            @test size(G, 1) == n
            @test size(G, 2) > 1
            approx = G * G'
            err = norm(K - approx, 1) / norm(K, 1)
            @test err < 1.0
        end

        @testset "rank cap on accepted pivots" begin
            # Set rank to a value smaller than block_size to trigger r > max_rank - k capping
            Random.seed!(42)
            G, pivots = rpcholesky(rbf, X, Val(true); rank=2, block_size=6)
            @test size(G, 2) <= 2
            @test length(pivots) == size(G, 2)
        end

        @testset "small final block (b < block_size)" begin
            Random.seed!(42)
            G, pivots = rpcholesky(rbf, X, Val(true); rank=7, block_size=4)
            @test size(G, 2) <= 7
            @test length(pivots) == size(G, 2)
        end

        @testset "type stability" begin
            @test (@inferred rpcholesky(rbf, X; rank=5)) isa Matrix{Float64}
            @test (@inferred rpcholesky(rbf, X, Val(true); rank=5)) isa Tuple{Matrix{Float64}, Vector{Int}}
        end
    end

end
