"""
Test fixed-point gradients through pulling-through contractions.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise

using Test
using Random
using PEPSKit
using TensorKit
using Zygote
using OptimKit
using KrylovKit

include("$(@__DIR__)/symmetrization.jl")

## Test models, gradmodes and CTMRG algorithm
# -------------------------------------------
χbond = 2
χenv = 20
Pspaces = [ComplexSpace(2), ComplexSpace(2)]
Vspaces = [ComplexSpace(χbond), ComplexSpace(χbond)]
Espaces = [ComplexSpace(χenv), ComplexSpace(χenv)]
models = [heisenberg_XYZ(InfiniteSquare()), transverse_field_ising(InfiniteSquare())]
names = ["Heisenberg", "Ising"]
symm = MyRotateReflect()
# no spatial symmetries for the p-wave superconductor, using Ising instead...

gradtol = 1e-4
pt_algs = [
    [PullingThrough(; tol=1e-10, verbosity=2, maxiter=500)],
    [PullingThrough(; tol=1e-10, verbosity=2, maxiter=500)],
]
gradmodes = [
    [
        # LinSolver(;
        #     solver=KrylovKit.GMRES(; tol=gradtol, maxiter=200, krylovdim=100),
        #     iterscheme=:square,
        # ),
        # LinSolver(;
        #     solver=KrylovKit.BiCGStab(; tol=gradtol, maxiter=200), iterscheme=:square
        # ),
        LSSolver(;
            solver=KrylovKit.LSMR(; tol=gradtol, maxiter=200, krylovdim=100),
            iterscheme=:rectangular,
        ),
    ],
    [
        # LinSolver(;
        #     solver=KrylovKit.GMRES(; tol=gradtol, maxiter=200, krylovdim=100),
        #     iterscheme=:square,
        # ),
        # LinSolver(;
        #     solver=KrylovKit.BiCGStab(; tol=gradtol, maxiter=200), iterscheme=:square
        # ),
        LSSolver(;
            solver=KrylovKit.LSMR(; tol=gradtol, maxiter=200, krylovdim=100),
            iterscheme=:rectangular,
        ),
    ],
]
steps = -0.01:0.005:0.01

## Tests
# ------
@testset "AD pulling-through energy gradients for $(names[i]) model" verbose = true for i in
                                                                                        eachindex(
    models
)
    Pspace = Pspaces[i]
    Vspace = Pspaces[i]
    Espace = Espaces[i]
    gms = gradmodes[i]
    ptalgs = pt_algs[i]
    @testset "$pt_alg and $alg_rrule" for (pt_alg, alg_rrule) in
                                          Iterators.product(ptalgs, gms)
        @info "optimtest of $pt_alg and $alg_rrule on $(names[i])"
        # Random.seed!(42039482030) # bad seed for pulling through it seems
        dir = InfinitePEPS(Pspace, Vspace, Vspace)
        psi = InfinitePEPS(Pspace, Vspace, Vspace)
        psi = symmetrize!(psi, symm)
        dir = symmetrize!(dir, symm)
        check_symmetry(psi, symm)
        check_symmetry(dir, symm)
        env, = leading_boundary(PullingThroughEnv(psi, Espace), psi, pt_alg)
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha=steps,
            retract=PEPSKit.peps_retract,
            inner=PEPSKit.real_inner,
        ) do (peps, envs)
            E, gs = Zygote.withgradient(peps) do psi
                env2, = PEPSKit.hook_pullback(leading_boundary, envs, psi, pt_alg; alg_rrule)
                return costfun(psi, env2, models[i])
            end
            # TODO: symmetrize the gradient here?
            g = only(gs)
            symmetrize!(g, symm)
            check_symmetry(g, symm)
            return E, g
        end
        @test dfs1 ≈ dfs2 atol = 1e-2
    end
end
