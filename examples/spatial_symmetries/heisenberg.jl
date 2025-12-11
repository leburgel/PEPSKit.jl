# incorporate spatial symmetries in PEPS optimization: example using Heisenberg model

using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit

Random.seed!(1234)

# Part O: Setup
# -------------

include(joinpath(@__DIR__, "spatial_toolbox.jl"))

# model
Jx = -1.0
Jy = 1.0
Jz = -1.0

# parameters
χbond = 3
χenv = 20

# algorithms
optim_maxiter = 100
gradient_iterscheme = :fixed
boundary_alg = SimultaneousCTMRG(;
    trunc = truncdim(χenv), tol = 1.0e-8, miniter = 3, maxiter = 400, verbosity = 2
)
gradient_alg = EigSolver(;
    solver_alg = Arnoldi(; tol = 1.0e-6, maxiter = 10, verbosity = 2, eager = true),
    iterscheme = gradient_iterscheme,
)
optimizer_alg = LBFGS(32; gradtol = 1.0e-5, verbosity = 3, maxiter = optim_maxiter)
reuse_env = true

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
H = heisenberg_XYZ(lattice(unitcell_style); Jx, Jy, Jz)

# spaces
P = first(H.lattice)
Vpeps = ℂ^χbond
Venv = ℂ^χenv

# Part I: manually imposing symmetries in gradient computation and retraction
# ---------------------------------------------------------------------------

## Initialization

# ititialize state and manually symmetrize
A0 = TensorMap(randn, ComplexF64, P ← Vpeps ⊗ Vpeps ⊗ Vpeps ⊗ Vpeps)
A0 = normalize(symmetrize(A0, symm_style))
ψ₀ = fill_peps(A0, unitcell_style)
env₀, = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

cfun, inner, retract, transport! = peps_opt_costfunction(
    H; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(A, env), f, g = optimize(
    cfun,
    (A0, env₀),
    optimizer_alg;
    inner = inner,
    retract = retract,
    (transport!) = (transport!),
);

# Part II: automatically impose spatial symmetries using spatially symmetric tensors
# ----------------------------------------------------------------------------------

## Initialization

A_basis = find_symmetric_basis(P, Vpeps, symm_style)
@assert all(≈(1.0), norm.(A_basis)) # make sure the basis is orthonormal
a₀ = normalize(randn(length(A_basis)))
ψ₀ = fill_peps(vec2peps(a₀, A_basis), unitcell_style)
env₀, = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

cfun, inner, retract, transport! = vector_opt_costfunction(
    H, P, Vpeps; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(a, env), f, g, numfg, history = optimize(
    cfun,
    (a₀, env₀),
    optimizer_alg;
    inner = inner,
    retract = retract,
    (transport!) = (transport!),
);

nothing
