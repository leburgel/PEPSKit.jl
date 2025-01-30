"""
Test properties and symmetries of a properly symmetrized pulling through environment.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise

using LinearAlgebra
using TensorKit
using MPSKit
using MPSKitModels
using PEPSKit
using KrylovKit
using OptimKit

include("$(@__DIR__)/symmetrization.jl")

# Square lattice Heisenberg Hamiltonian
# We use the parameters (J₁, J₂, J₃) = (-1, 1, -1) by default to capture
# the ground state in a single-site unit cell. This can be seen from
# sublattice rotating H from parameters (1, 1, 1) to (-1, 1, -1).
H = heisenberg_XYZ(InfiniteSquare(); Jx=-1, Jy=1, Jz=-1)

# Parameters
χbond = 2 # TODO: play around with this...
χenv = 12 # TODO: can't use too large environment bond dimensions for small PEPS bond dimensions/very gapped states?
symm = MyRotateReflect()
pt_alg = PullingThrough(; tol=1e-10, verbosity=2, maxiter=500)
opt_alg = PEPSOptimize(;
    boundary_alg=pt_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=3, ls_maxiter=2, ls_maxfg=4),
    gradient_alg=LinSolver(;
        solver=KrylovKit.GMRES(;
            maxiter=30,
            tol=PEPSKit.Defaults.fpgrad_tol,
            verbosity=2,
        ),
        iterscheme=:real, # toggle between :real and :complex
    ),
    reuse_env=true,
)

symm_tol = 1e-10
function my_finalize!((peps, envs), f, g, numiter)
    println("Checking symmetries...")
    check_symmetry(peps, symm; tol=symm_tol)
    check_symmetry(g, symm; tol=symm_tol)
    return (peps, envs), f, g
end

# Ground state search
# We initialize a random PEPS with bond dimension χbond and from that converge
# a CTMRG environment with dimension χenv on the environment bonds before
# starting the optimization. The ground-state energy should approximately approach
# E/N = −0.6694421, which is a QMC estimate from https://arxiv.org/abs/1101.3281.
# Of course there is a noticable bias for small χbond and χenv.

# Start from random PEPS: not really contracting very well...
ψ₀ = InfinitePEPS(ℂ^2, ℂ^χbond)
ψ₀ = symmetrize!(ψ₀, symm)
ψ₀[1, 1] /= norm(ψ₀[1, 1], Inf)

# # EXTRA STEP: seed with simple update, gives trivial initial contraction but then things still go wrong
# ψ₀ = InfiniteWeightPEPS(rand, Float64, ℂ^2, ℂ^χbond)
# ψ₀.vertices[1, 1] /= norm(ψ₀.vertices[1, 1], Inf)

# # simple update
# dts = [1e-2, 1e-3, 4e-4, 1e-4]
# tols = [1e-6, 1e-8, 1e-8, 1e-8]
# maxiter = 10000
# for (n, (dt, tol)) in enumerate(zip(dts, tols))
#     trscheme = truncerr(1e-10) & truncdim(χbond)
#     alg = SimpleUpdate(dt, tol, maxiter, trscheme)
#     global ψ₀, = simpleupdate(ψ₀, H, alg; bipartite=false)
# end
# # absorb weight into site tensors
# ψ₀ = InfinitePEPS(ψ₀)

env₀, N, ϵ = leading_boundary(PullingThroughEnv(ψ₀, ℂ^χenv), ψ₀, pt_alg)
ψ₀ = ψ₀ / sqrt(N)

result = fixedpoint(ψ₀, H, opt_alg, env₀; symmetrization=symm, (finalize!)=(my_finalize!))
@show result.E

nothing
