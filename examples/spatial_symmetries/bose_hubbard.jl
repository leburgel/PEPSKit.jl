# incorporate spatial symmetries in PEPS optimization: example using Heisenberg model

using TensorKit
using PEPSKit
import MPSKit: add_physical_charge
using OptimKit
using KrylovKit

# Part O: Setup
# -------------

include(joinpath(@__DIR__, "spatial_toolbox.jl"))
include(joinpath(@__DIR__, "u1_toolbox.jl"))

# spaces
symmetry = U1Irrep
Vpeps = U1Space(0 => 2, 1 => 1, -1 => 1)
Venv = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2)

# uniform auxiliary physical charge, for uniform half-filling
Saux = U1Irrep(-1)

# parameters
t = 1.0
U = 20.0
cutoff = 2

# algorithms
optim_maxiter = 100
boundary_alg = SimultaneousCTMRG(;
    trscheme=FixedSpaceTruncation(), tol=1e-10, miniter=3, maxiter=400, verbosity=2
)
gradient_alg = EigSolver(;
    solver_alg=Arnoldi(; tol=1e-6, maxiter=30, verbosity=2, eager=true),
    iterscheme=:diffgauge,
) # :diffgauge necessary for :sequential CTMRG scheme
optimizer_alg = LBFGS(; gradtol=1e-4, verbosity=3, maxiter=optim_maxiter)
reuse_env = true

# choose symmetrization style: needs to be U1-compatible!
# symm_style = None()
# symm_style = Rotation()
symm_style = U1HReflectionRotation() # so we need charge conjugation to make things match!

# choose unit cell style: still uniform so can just use the same filling styles as before
unitcell_style = Asymmetric()
# unitcell_style = Symmetric()

# shifted square lattice Bose-Hubbard Hamiltonian
H0 = bose_hubbard_model(symmetry, lattice(unitcell_style); t, U, cutoff)
H = add_physical_charge(H0, fill(Saux, size(H0.lattice)))
P = first(H.lattice) # uniform physical space

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
