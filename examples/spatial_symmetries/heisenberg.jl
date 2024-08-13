# incorporate spatial symmetries in PEPS optimization: example using Heisenberg model

using LinearAlgebra
using KrylovKit
using TensorKit
using PEPSKit
using Zygote
using ChainRulesCore
using OptimKit

using PEPSKit: PEPSTensor

# Part O: Setup
# -------------

include("spatial_toolbox.jl")

# parameters
χbond = 3
χenv = 20
boundary_alg = CTMRG(;
    trscheme=truncdim(χenv), tol=1e-10, miniter=3, maxiter=400, verbosity=2, ctmrgscheme=:sequential,
)
gradient_alg = LinSolver(; solver=GMRES(; tol=1e-6, maxiter=10, verbosity=2), iterscheme=:diffgauge) # :diffgauge necessary for :sequential CTMRG scheme
optimization_alg = LBFGS(; gradtol=1e-4, verbosity=2)
reuse_env = true
verbosity = 2

# choose symmetrization style
# symm_style = None() # no spatial symmetries
# symm_style = Rotation() # rotation invariance
# symm_style = Reflection() # reflection invariance
# symm_style = ReflectionRotation() # reflection and rotation invariance
# symm_style = HReflection() # Hermitian reflection invariance
symm_style = HReflectionRotation() # rotation and Hermitian reflection invariance

# choose unit cell style
unitcell_style = Asymmetric()
# unitcell_style = Symmetric()

# square lattice Heisenberg Hamiltonian
heisenberg_ham(::Asymmetric) = square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
heisenberg_ham(::Symmetric) = repeat(heisenberg_ham(Asymmetric()), 2, 2)

H = heisenberg_ham(unitcell_style)

# spaces
P = ℂ^2
Vpeps = ℂ^χbond
Venv = ℂ^χenv

# Part I: manually imposing symmetries in gradient computation and retraction
# ---------------------------------------------------------------------------

## Initialization

# ititialize state and manually symmetrize
A0 = TensorMap(randn, ComplexF64, P ← Vpeps ⊗ Vpeps ⊗ Vpeps ⊗ Vpeps)
A0 = symmetrize(A0, symm_style)
ψ₀ = fill_peps(A0, unitcell_style)
env₀ = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

peps_cfun, peps_retract, peps_inner = peps_opt_costfunction(;
    boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(A, env), f, g = optimize(
    peps_cfun, (A0, env₀), optimization_alg; retract=peps_retract, inner=peps_inner
);

# Part II: automatically impose spatial symmetries using spatially symmetric tensors
# ----------------------------------------------------------------------------------

## Initialization

A_basis = find_symmetric_basis(P, Vpeps, symm_style)
a₀ = randn(length(A_basis))
ψ₀ = fill_peps(vec2peps(a₀, A_basis), unitcell_style)
env₀ = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

vector_cfun, vector_retract, vector_inner = vector_opt_costfunction(
    P, Vpeps; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(a, env), f, g, numfg, history = optimize(
    vector_cfun, (a₀, env₀), optimization_alg; retract=vector_retract, inner=vector_inner
);

nothing


# D = 3: non-symm converges to E = -0.663... (??? no clue cause it never converges); symm converges to E = -0.66756...
