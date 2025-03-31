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

using Logging
using LoggingExtras

file_logger = MinLevelLogger(FileLogger("logfile.txt"), Logging.Info)
global_logger(file_logger)

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

boundary_tol = 1e-12
boundary_maxiter = 500
boundary_verbosity = 2
boundary_gauge = :center

fpgrad_style = :naive
fpgrad_tol = 1e-8

pt_algs = [
    [
        PullingThrough(;
            tol=boundary_tol,
            verbosity=boundary_verbosity,
            maxiter=boundary_maxiter,
            gauge=boundary_gauge,
        ),
    ],
    [
        PullingThrough(;
            tol=boundary_tol,
            verbosity=boundary_verbosity,
            maxiter=boundary_maxiter,
            gauge=boundary_gauge,
        ),
    ],
]
gradmodes = [
    [
        PTLSSolver(;
            solver_alg=KrylovKit.LSMR(;
                tol=fpgrad_tol, maxiter=1000, krylovdim=1000, verbosity=2
            ),
            gauge=boundary_gauge,
            style=fpgrad_style,
        ),
    ],
    [
        PTLSSolver(;
            solver_alg=KrylovKit.LSMR(;
                tol=fpgrad_tol, maxiter=1000, krylovdim=1000, verbosity=2
            ),
            gauge=boundary_gauge,
            style=fpgrad_style,
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
                return cost_function(psi, env2, models[i])
            end
            g = only(gs)
            symmetrize!(g, symm)
            check_symmetry(g, symm)
            return E, g
        end
        @test dfs1 ≈ dfs2 atol = 1e-2
    end
end
