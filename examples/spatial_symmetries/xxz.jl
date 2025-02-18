# exploiting charge conjugation and spatial symmetries in the XXZ Heisenberg model

using KrylovKit
using TensorKit
using PEPSKit
using OptimKit

include("spatial_toolbox.jl")
include("u1_toolbox.jl")
include("space_shifting.jl")

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
boundary_alg = SimultaneousCTMRG(;
    trscheme=FixedSpaceTruncation(), tol=1e-10, miniter=3, maxiter=100, verbosity=2
)
gradient_alg = LinSolver(;
    solver=GMRES(; tol=1e-6, maxiter=10, verbosity=2), iterscheme=:diffgauge
) # :diffgauge necessary for :sequential CTMRG scheme
optimization_alg = LBFGS(; gradtol=1e-4, verbosity=3)
reuse_env = true

# virtual spaces
Nspaces = [Vpeps Vpeps; Vpeps Vpeps]
Espaces = [Vpeps Vpeps; Vpeps Vpeps]

# shift Hamiltonian and record shifted physical spaces
H1 = heisenberg_XXZ(ComplexF64, U1Irrep, InfiniteSquare(2, 2); J=1.0, Δ=1.0, spin=1//2)
H, Pspaces = add_physical_charge(H1, Saux)

# Part I: naive optimization using a 2-site unit cell
# ---------------------------------------------------

mode = "naive optimization with 2x2 unit cell"

@info "Running $mode"

## Initialization

Nspaces = [Vpeps Vpeps; Vpeps Vpeps]
Espaces = [Vpeps Vpeps; Vpeps Vpeps]
ψ₀ = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
env₀ = CTMRGEnv(ψ₀, Venv)
env₀, = leading_boundary(env₀, ψ₀, boundary_alg)

## Optimize

pepsopt_alg = PEPSOptimize(;
    boundary_alg=boundary_alg,
    optimizer=optimization_alg,
    gradient_alg=gradient_alg,
    reuse_env=reuse_env,
)
result = fixedpoint(H, ψ₀, env₀, pepsopt_alg)

@info "Finished $mode"

numfg = result.numfg
E = result.E

@info "Energy: $E\t numfg: $numfg\t numiter: ???"

# Part II: spatial and charge conjugation symmetry, trivial flipper
# -----------------------------------------------------------------

mode = "spatial and charge-conjugation symmetry, using trivial flipper"

@info "Running $mode"

# cannot use manifestly spatially symmetric tensors when also imposing U1 charge conjugation...

## Setup

# symm_style = Rotation()
# symm_style = U1HReflection()
symm_style = U1HReflectionRotation()

unitcell_style = U1Symmetric()

## Initialization

A0 = TensorMap(randn, ComplexF64, Pspaces[1, 1] ← Vpeps ⊗ Vpeps ⊗ Vpeps ⊗ Vpeps)
A0 = symmetrize(A0, symm_style)
ψ₀ = fill_peps(A0, unitcell_style)
env₀, = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

peps_cfun, peps_retract, peps_inner = peps_opt_costfunction(;
    boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(A, env), f, g, numfg, history = optimize(
    peps_cfun, (A0, env₀), optimization_alg; retract=peps_retract, inner=peps_inner
);

@info "Finished $mode"

@info "Energy: $E\t numfg: $numfg\t numiter: $(length(history[2]))"

# Part III: spatial and charge conjugation symmetry, NONTRIVIAL FLIPPER -> WORKING
# ---------------------------------------------------------------------

mode = "spatial and charge-conjugation symmetry, using NON-TRIVIAL flipper"

@info "Running $mode"

# cannot use manifestly spatially symmetric tensors when also imposing U1 charge conjugation?

## Setup

# symm_style = Rotation()
# symm_style = U1XHReflection()
symm_style = U1XHReflectionRotation()

unitcell_style = U1XSymmetric()

## Initialization

A0 = TensorMap(randn, ComplexF64, Pspaces[1, 1] ← Vpeps ⊗ Vpeps ⊗ Vpeps ⊗ Vpeps)
A0 = symmetrize(A0, symm_style)
ψ₀ = fill_peps(A0, unitcell_style)
env₀, = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

peps_cfun, peps_retract, peps_inner = peps_opt_costfunction(;
    boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(A, env), f, g, numfg, history = optimize(
    peps_cfun, (A0, env₀), optimization_alg; retract=peps_retract, inner=peps_inner
);

@info "Finished $mode"

@info "Energy: $E\t numfg: $numfg\t numiter: $(length(history[2]))"
