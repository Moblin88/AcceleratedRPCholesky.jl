using Test
using LinearAlgebra
using Random
using AcceleratedRPCholesky

Random.seed!(42)

@testset "AcceleratedRPCholesky.jl" begin

    @testset "rpcholesky - kernel interface" begin
        n = 15
        d = 3
        X = randn(n, d)

        rbf(x, y) = exp(-sum((x .- y).^2) / 2.0)

        K = [rbf(X[i,:], X[j,:]) for i in 1:n, j in 1:n]

        @testset "returns a matrix" begin
            G = rpcholesky(rbf, X; rank=5)
            @test G isa Matrix
            @test size(G, 1) == n
            @test size(G, 2) <= 5
        end

        @testset "approximation quality" begin
            G = rpcholesky(rbf, X; rank=n, rtol=1e-6)
            approx = G * G'
            err = norm(K - approx, 1) / norm(K, 1)
            @test err < 0.5
        end

        @testset "dynamic rank (defaults to n)" begin
            G = rpcholesky(rbf, X; rtol=0.05)
            @test size(G, 1) == n
        end

        @testset "block_size parameter accepted" begin
            G = rpcholesky(rbf, X; rank=3, block_size=2)
            @test size(G, 2) <= 3
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

        @testset "G capacity growth beyond block_size" begin
            # Use block_size=1 and rank > block_size to force G reallocation:
            # the initial G has 1 column; after the first accepted pivot k=1,
            # the next iteration triggers the k+r > size(G,2) growth branch.
            Random.seed!(42)
            G = rpcholesky(rbf, X; rank=8, block_size=1)
            @test size(G, 1) == n
            @test size(G, 2) > 1  # confirms reallocation path was exercised
            approx = G * G'
            err = norm(K - approx, 1) / norm(K, 1)
            @test err < 1.0
        end

        @testset "rank cap on accepted pivots" begin
            # Set rank to a value smaller than block_size to trigger r > max_rank - k capping
            Random.seed!(42)
            G = rpcholesky(rbf, X; rank=2, block_size=6)
            @test size(G, 2) <= 2
        end

        @testset "small final block (b < block_size)" begin
            # block_size=4, rank=7: last block has a partial remainder of 3, hits the fresh-matrix branch
            Random.seed!(42)
            G = rpcholesky(rbf, X; rank=7, block_size=4)
            @test size(G, 2) <= 7
        end
    end

end
