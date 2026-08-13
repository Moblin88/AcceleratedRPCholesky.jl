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
    end

end
