using Test
using LinearAlgebra
using Random
using RPCholesky

Random.seed!(42)

@testset "RPCholesky.jl" begin

    @testset "rpcholesky - matrix interface" begin
        n = 20
        A_raw = randn(n, n)
        A = A_raw * A_raw' + n * I  # well-conditioned PSD

        @testset "returns RPCholeskyResult" begin
            F = rpcholesky(A; rank=5)
            @test F isa RPCholeskyResult
        end

        @testset "low rank approximation quality" begin
            F = rpcholesky(A; rank=n, rtol=1e-6)
            approx = F.G * F.G'
            err = norm(A - approx) / norm(A)
            @test err < 0.5
        end

        @testset "rank constraint respected" begin
            r = 4
            F = rpcholesky(A; rank=r)
            @test F.rank <= r
        end

        @testset "rtol stopping" begin
            F_full = rpcholesky(A; rank=n, rtol=0.0)
            F_rtol = rpcholesky(A; rank=n, rtol=0.5)
            @test F_rtol.rank <= F_full.rank
        end

        @testset "block_size parameter accepted" begin
            F = rpcholesky(A; rank=5, block_size=2)
            @test F isa RPCholeskyResult
        end

        @testset "rank defaults to n" begin
            F = rpcholesky(A; rtol=1e-10)
            @test size(F.G, 1) == n
        end

        @testset "piv contains pivot indices" begin
            F = rpcholesky(A; rank=5)
            @test length(F.piv) == F.rank
            @test all(1 .<= F.piv .<= n)
        end
    end

    @testset "rpcholesky_kernel - kernel interface" begin
        n = 15
        d = 3
        X = randn(n, d)

        rbf(x, y) = exp(-sum((x .- y).^2) / 2.0)

        K = [rbf(X[i,:], X[j,:]) for i in 1:n, j in 1:n]

        @testset "returns a matrix" begin
            L = rpcholesky_kernel(rbf, X; rank=5)
            @test L isa Matrix
            @test size(L, 1) == n
            @test size(L, 2) <= 5
        end

        @testset "approximation quality" begin
            L = rpcholesky_kernel(rbf, X; rank=n, rtol=1e-6)
            approx = L * L'
            err = norm(K - approx, 1) / norm(K, 1)
            @test err < 0.5
        end

        @testset "dynamic rank (defaults to n)" begin
            L = rpcholesky_kernel(rbf, X; rtol=0.05)
            @test size(L, 1) == n
        end

        @testset "block_size parameter accepted" begin
            L = rpcholesky_kernel(rbf, X; rank=3, block_size=2)
            @test size(L, 2) <= 3
        end

        @testset "rtol stopping" begin
            L_tight = rpcholesky_kernel(rbf, X; rank=n, rtol=0.01)
            L_loose = rpcholesky_kernel(rbf, X; rank=n, rtol=0.9)
            @test size(L_loose, 2) <= size(L_tight, 2)
        end
    end

    @testset "edge cases" begin
        @testset "rank-1 PSD matrix" begin
            v = [1.0, 2.0, 3.0]
            A = v * v'
            F = rpcholesky(A; rank=1, rtol=0.0)
            @test F.rank >= 1
            @test norm(A - F.G * F.G') / norm(A) < 0.01
        end

        @testset "identity matrix full rank" begin
            A = Matrix{Float64}(I, 5, 5)
            F = rpcholesky(A; rtol=1e-10)
            approx = F.G * F.G'
            @test norm(A - approx) / norm(A) < 0.1
        end

        @testset "2x2 PSD matrix" begin
            A = [4.0 2.0; 2.0 3.0]
            F = rpcholesky(A; rank=2, rtol=0.0)
            @test norm(A - F.G * F.G') / norm(A) < 0.01
        end
    end
end
