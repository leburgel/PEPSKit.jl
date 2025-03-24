# incorporate spatial symmetries in PEPS optimization: example using Heisenberg model

using TensorKit
using PEPSKit
using KrylovKit
using OptimKit

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
gradient_iterscheme = :diffgauge
boundary_alg = SimultaneousCTMRG(;
    trscheme=truncdim(χenv), tol=1e-8, miniter=3, maxiter=400, verbosity=2
)
# iterscheme=:fixed is giving me svdsolve cotangent issues, and some errors too...
gradient_alg = EigSolver(;
    solver_alg=Arnoldi(; tol=1e-6, maxiter=10, verbosity=2, eager=true),
    iterscheme=gradient_iterscheme,
) # :diffgauge necessary for :sequential CTMRG scheme
optimizer_alg = LBFGS(; gradtol=1e-4, verbosity=3, maxiter=optim_maxiter)
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

peps_cfun, peps_inner, peps_retract, peps_transport! = peps_opt_costfunction(
    H; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(A, env), f, g = optimize(
    peps_cfun,
    (A0, env₀),
    optimizer_alg;
    inner=peps_inner,
    retract=peps_retract,
    (transport!)=(peps_transport!),
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

vector_cfun, vector_inner, vector_retract, vector_transport! = vector_opt_costfunction(
    H, P, Vpeps; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(a, env), f, g, numfg, history = optimize(
    vector_cfun,
    (a₀, env₀),
    optimizer_alg;
    inner=vector_inner,
    retract=vector_retract,
    (transport!)=(vector_transport!),
);

nothing

# D = 3: non-symm converges to E = -0.663... (??? no clue cause it never converges); symm converges to E = -0.66756...
