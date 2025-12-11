# exploiting charge conjugation and spatial symmetries in the XXZ Heisenberg model

using Random
using KrylovKit
using TensorKit
using PEPSKit
using OptimKit
using MPSKit: add_physical_charge

include(joinpath(@__DIR__, "spatial_toolbox.jl"))
include(joinpath(@__DIR__, "u1_toolbox.jl"))

Random.seed!(1234)

# Part 0: Setup
# -------------

# spaces
Vpeps = U1Space(0 => 2, 1 => 1, -1 => 1) # should get me somewhere close to E = -0.669...?
Venv = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2)
# staggered auxiliary physical charges
Saux = [
    U1Irrep(-1 // 2) U1Irrep(1 // 2)
    U1Irrep(1 // 2) U1Irrep(-1 // 2)
]

# parameters
χenv = 18
optim_maxiter = 100
gradient_iterscheme = :fixed
boundary_alg = SimultaneousCTMRG(;
    trunc = FixedSpaceTruncation(), tol = 1.0e-8, miniter = 3, maxiter = 100, verbosity = 2
)
gradient_alg = EigSolver(;
    solver_alg = Arnoldi(; tol = 1.0e-6, maxiter = 10, verbosity = 1, eager = true),
    iterscheme = gradient_iterscheme,
)
optimizer_alg = LBFGS(32; gradtol = 1.0e-4, verbosity = 3, maxiter = optim_maxiter)
reuse_env = true

# shift Hamiltonian and record shifted physical spaces
H0 = heisenberg_XXZ(ComplexF64, U1Irrep, InfiniteSquare(2, 2); J = 1.0, Delta = 1.0, spin = 1 // 2)
H = add_physical_charge(H0, Saux)
Pspaces = H.lattice

# # Part I: naive optimization using a 2-site unit cell
# # ---------------------------------------------------

# mode = "naive optimization with 2x2 unit cell"

# @info "Running $mode"

# ## Initialization

# Nspaces = [Vpeps Vpeps; Vpeps Vpeps]
# Espaces = [Vpeps Vpeps; Vpeps Vpeps]
# ψ₀ = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
# env₀ = CTMRGEnv(ψ₀, Venv)
# env₀, = leading_boundary(env₀, ψ₀, boundary_alg)

# ## Optimize

# pepsopt_alg = PEPSOptimize(; boundary_alg, optimizer_alg, gradient_alg, reuse_env)
# ψ, env, E, info = fixedpoint(H, ψ₀, env₀, pepsopt_alg)

# @info "Finished $mode"

# numfg = info.fg_evaluations
# numiter = length(info.costs)

# @info "Energy: $E\t numfg: $numfg\t numiter: $numiter"

# Part II: spatial and charge conjugation symmetry, trivial flip
# --------------------------------------------------------------

mode = "spatial and charge-conjugation symmetry, using trivial flipper"

@info "Running $mode"

# cannot use manifestly spatially symmetric tensors with a staggered symmetry

## Setup

# symm_style = None()
# symm_style = Rotation()
# symm_style = U1HReflection()
symm_style = U1HReflectionRotation()

unitcell_style = U1Symmetric()

## Initialization

A0 = TensorMap(randn, ComplexF64, Pspaces[1, 1] ← Vpeps ⊗ Vpeps ⊗ Vpeps ⊗ Vpeps)
A0 = normalize(symmetrize(A0, symm_style))
ψ₀ = fill_peps(A0, unitcell_style)
env₀, = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

peps_cfun, peps_inner, peps_retract, peps_transport! = peps_opt_costfunction(
    H; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(A, env), f, g, numfg, history = optimize(
    peps_cfun,
    (A0, env₀),
    optimizer_alg;
    inner = peps_inner,
    retract = peps_retract,
    (transport!) = (peps_transport!),
);

@info "Finished $mode"

@info "Energy: $f\t numfg: $numfg\t numiter: $(length(history[2]))"

# Part III: spatial and charge conjugation symmetry, non-trivial flip
# -------------------------------------------------------------------

mode = "spatial and charge-conjugation symmetry, using non-trivial flipper"

@info "Running $mode"

# cannot use manifestly spatially symmetric tensors when also imposing U1 charge conjugation?

## Setup

# symm_style = Rotation()
# symm_style = U1XHReflection()
symm_style = U1XHReflectionRotation()

unitcell_style = U1XSymmetric() # non-trivial flipper

## Initialization

A0 = TensorMap(randn, ComplexF64, Pspaces[1, 1] ← Vpeps ⊗ Vpeps ⊗ Vpeps ⊗ Vpeps)
A0 = normalize(symmetrize(A0, symm_style))
ψ₀ = fill_peps(A0, unitcell_style)
env₀, = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

peps_cfun, peps_inner, peps_retract, peps_transport! = peps_opt_costfunction(
    H; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(A, env), f, g, numfg, history = optimize(
    peps_cfun,
    (A0, env₀),
    optimizer_alg;
    inner = peps_inner,
    retract = peps_retract,
    (transport!) = (peps_transport!),
);

@info "Finished $mode"

@info "Energy: $f\t numfg: $numfg\t numiter: $(length(history[2]))"
