using Test
using Random
using PEPSKit
using TensorKit

using PEPSKit: ctmrg_iter, gauge_fix, check_elementwise_convergence

scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 2)]
χ = Dict([(1, 1) => 8, (2, 2) => 26, (3, 2) => 26])  # Increase χ to converge non-symmetric environments

@testset "Trivial symmetry ($T) - ($unitcell)" for (T, unitcell) in
                                                   Iterators.product(scalartypes, unitcells)
    physical_space = ComplexSpace(2)
    peps_space = ComplexSpace(2)
    ctm_space = ComplexSpace(χ[unitcell])

    Random.seed!(2938293852938)  # Seed RNG to make random environment consistent
    # Random.seed!(928329384)  # Seed RNG to make random environment consistent
    psi = InfinitePEPS(randn, T, physical_space, peps_space; unitcell)
    ctm = CTMRGEnv(psi; Venv=ctm_space)

    alg = CTMRG(;
        tol=1e-10,
        maxiter=100,
        verbosity=1,
        trscheme=FixedSpaceTruncation(),
        ctmrgscheme=:AllSides,  # In general :AllSides is faster
    )

    ctm = leading_boundary(ctm, psi, alg)
    ctm2, = ctmrg_iter(psi, ctm, alg)
    ctm_fixed, = gauge_fix(ctm, ctm2)
    @test PEPSKit.check_elementwise_convergence(ctm, ctm_fixed; atol=1e-6)
end

@testset "Z2 symmetry ($T) - ($unitcell)" for (T, unitcell) in
                                              Iterators.product(scalartypes, unitcells)
    physical_space = Z2Space(0 => 1, 1 => 1)
    peps_space = Z2Space(0 => 1, 1 => 1)
    ctm_space = Z2Space(0 => χ[(1, 1)] ÷ 2, 1 => χ[(1, 1)] ÷ 2)

    Random.seed!(2938293852938)  # Seed RNG to make random environment consistent
    psi = InfinitePEPS(physical_space, peps_space; unitcell)
    ctm = CTMRGEnv(psi; Venv=ctm_space)

    alg = CTMRG(;
        tol=1e-10,
        maxiter=400,
        verbosity=1,
        trscheme=FixedSpaceTruncation(),
        ctmrgscheme=:LeftMoves,  # Weirdly, only :LeftMoves converges
    )

    ctm = leading_boundary(ctm, psi, alg)
    ctm2, = ctmrg_iter(psi, ctm, alg)
    ctm_fixed, = gauge_fix(ctm, ctm2)
    @test PEPSKit.check_elementwise_convergence(ctm, ctm_fixed; atol=1e-6)
end
