using Test
using LinearAlgebra
using Random
using DataFrames
using AcceleratedRPCholesky

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

        Random.seed!(42)
        L, accepted = AcceleratedRPCholesky._block_pivot_cholesky!(
            H,
            size(H, 1),
            1e-12,
        )

        @test !isempty(accepted)
        @test H != H0
        @test Matrix(L * L') ≈ H0[accepted, accepted]
    end

    @testset "rpcholesky - kernel interface" begin
        n = 15
        d = 3
        X = randn(n, d)

        rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2.0)

        K = [rbf(X[i,:], X[j,:]) for i in 1:n, j in 1:n]

        @testset "returns a matrix" begin
            G, pivots = rpcholesky(rbf, X; rank=5)
            @test G isa Matrix
            @test size(G, 1) == n
            @test size(G, 2) <= 5
            @test pivots isa Vector
            @test length(pivots) == size(G, 2)
        end

        @testset "supports DataFrames" begin
            data = DataFrame(x=X[:, 1], y=X[:, 2], z=X[:, 3])
            dataframe_rbf(x, y) = exp(-sum((xi - yi)^2 for (xi, yi) in zip(x, y)) / 2.0)
            G, pivots = rpcholesky(dataframe_rbf, data; rank=5)
            @test G isa Matrix{Float64}
            @test size(G, 1) == n
            @test size(G, 2) <= 5
            @test length(pivots) == size(G, 2)
        end

        @testset "preserves Float32 kernel type" begin
            X32 = Float32.(X)
            rbf32(x, y) = exp(Float32(-sum((x .- y).^2) / 2))
            G, pivots = rpcholesky(rbf32, X32; rank=5)
            @test G isa Matrix{Float32}
            @test size(G, 1) == n
            @test size(G, 2) <= 5
            @test eltype(G * G') == Float32
            @test length(pivots) == size(G, 2)
        end

        @testset "approximation quality" begin
            G, pivots = rpcholesky(rbf, X; rank=n, rtol=1e-6)
            approx = G * G'
            err = norm(K - approx, 1) / norm(K, 1)
            @test err < 0.5
        end

        @testset "keyword rank defaults to min(n, 50)" begin
            X_default = randn(51, d)
            G, pivots = rpcholesky(rbf, X_default)
            @test size(G, 1) == size(X_default, 1)
            @test size(G, 2) <= 50
            @test length(pivots) == size(G, 2)
        end

        @testset "block_size parameter accepted" begin
            G, pivots = rpcholesky(rbf, X; rank=3, block_size=2)
            @test size(G, 2) <= 3
            @test length(pivots) == size(G, 2)
        end

        @testset "rtol stopping" begin
            Random.seed!(42)
            G_tight, _ = rpcholesky(rbf, X; rank=n, rtol=0.01)
            Random.seed!(42)
            G_loose, _ = rpcholesky(rbf, X; rank=n, rtol=0.9)
            @test size(G_loose, 2) <= size(G_tight, 2)
        end

        @testset "residual trace tolerance" begin
            Random.seed!(42)
            rtol = 0.2
            G, pivots = rpcholesky(rbf, X; rank=n, rtol=rtol, block_size=4)
            residual = K - G * G'
            @test tr(Symmetric(residual)) <= rtol * tr(Symmetric(K)) + 1e-8
        end

        @testset "zero kernel returns empty factor" begin
            zero_kernel(x, y) = 0.0
            G, pivots = rpcholesky(zero_kernel, X; rank=5)
            @test G isa Matrix
            @test size(G, 2) == 0
            @test length(pivots) == 0
        end

        @testset "rank greater than block size" begin
            Random.seed!(42)
            G, pivots = rpcholesky(rbf, X; rank=8, block_size=1)
            @test size(G, 1) == n
            @test size(G, 2) > 1
            @test length(pivots) == size(G, 2)
            approx = G * G'
            err = norm(K - approx, 1) / norm(K, 1)
            @test err < 1.0
        end

        @testset "rank cap on accepted pivots" begin
            # Set rank to a value smaller than block_size to trigger r > max_rank - k capping
            Random.seed!(42)
            G, pivots = rpcholesky(rbf, X; rank=2, block_size=6)
            @test size(G, 2) <= 2
            @test length(pivots) == size(G, 2)
        end

        @testset "small final block (b < block_size)" begin
            Random.seed!(42)
            G, pivots = rpcholesky(rbf, X; rank=7, block_size=4)
            @test size(G, 2) <= 7
            @test length(pivots) == size(G, 2)
        end
    end

end
