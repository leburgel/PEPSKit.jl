# exploiting(?) charge conjugation and spatial symmetries in the XXZ Heisenberg model

using LinearAlgebra
using KrylovKit
using TensorOperations
using TensorKit
using MPSKit
using PEPSKit
using OptimKit
using ChainRulesCore
using Zygote
using PEPSKit:
    PEPSTensor, NORTHWEST, NORTHEAST, SOUTHEAST, SOUTHWEST, NORTH, EAST, SOUTH, WEST
using MPSKitModels: S_plusmin, S_minplus, S_zz

include("spatial_toolbox.jl")
include("u1_toolbox.jl")

# Part 0: Setup
# -------------

# spaces
Vpeps = U1Space(0 => 2, 1 => 1, -1 => 1) # should get me somewhere close to E = -0.669...?
Venv = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2)
P = U1Space(1 / 2 => 1, -1 / 2 => 1)
# staggered auxiliary physical spaces
Paux = [U1Space(-1 / 2 => 1) U1Space(1 / 2 => 1); U1Space(1 / 2 => 1) U1Space(-1 / 2 => 1)] 
# fuse auxiliary spaces with physical spaces
Pspaces = map(Paux) do P´
    fuse(P, P´)
end
Nspaces = [Vpeps Vpeps; Vpeps Vpeps]
Espaces = [Vpeps Vpeps; Vpeps Vpeps]

# parameters
χenv = 18
boundary_alg = CTMRG(;
    trscheme=FixedSpaceTruncation(), tol=1e-10, miniter=3, maxiter=100, verbosity=1, ctmrgscheme=:sequential,
)
gradient_alg = LinSolver(; solver=GMRES(; tol=1e-6, maxiter=10, verbosity=2), iterscheme=:diffgauge) # :diffgauge necessary for :sequential CTMRG scheme
optimization_alg = LBFGS(; gradtol=1e-4, verbosity=2)
reuse_env = true
verbosity = 2

# Heisenberg Hamiltonian without explicit U1 shift
function square_lattice_heisenberg(; J=1.0, Δ=1.0, spin=1//2)
    H =
        J * (
            (S_plusmin(U1Irrep; spin=spin) + S_minplus(U1Irrep; spin=spin)) / 2 +
            Δ * S_zz(U1Irrep; spin=spin)
        )
    return H / 4
end

# Heisenberg Hamiltonian with explicit U1 shift
function mod_square_lattice_heisenberg(; J=1.0, Δ=1.0, spin=1//2)
    H = square_lattice_heisenberg(; J, Δ, spin)
    I1 = id(Paux[1, 1])
    I2 = id(Paux[1, 2])
    H´ = H ⊗ I1 ⊗ I2
    f1 = isomorphism(fuse(space(H´, 1), space(H´, 3)), space(H´, 1) ⊗ space(H´, 3))
    f2 = isomorphism(fuse(space(H´, 2), space(H´, 4)), space(H´, 2) ⊗ space(H´, 4))
    @tensor H_AB[-1 -2; -3 -4] :=
        H´[1 2 3 4; 5 6 7 8] *
        f1[-1; 1 3] *
        f2[-2; 2 4] *
        conj(f1[-3; 5 7]) *
        conj(f2[-4; 6 8])
    H_BA = permute(H_AB, ((2, 1), (4, 3)))

    terms = []
    for (r, c) in Iterators.product(1:2, 1:2)
        H = mod(r + c, 2) == 0 ? H_AB : H_BA
        println()
        push!(terms, (CartesianIndex(r, c), CartesianIndex(r, c + 1)) => H)
        push!(terms, (CartesianIndex(r, c), CartesianIndex(r + 1, c)) => H)
    end

    return LocalOperator(Pspaces, terms...)
end

H = mod_square_lattice_heisenberg(; J=1.0, Δ=1.0, spin=1//2) # fused

# Part I: naive optimization using a 2-site unit cell
# ---------------------------------------------------

mode = "naive optimization with 2x2 unit cell"

@info "Running $mode"

## Initialization

Nspaces = [Vpeps Vpeps; Vpeps Vpeps]
Espaces = [Vpeps Vpeps; Vpeps Vpeps]
ψ₀ = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
env₀ = CTMRGEnv(ψ₀, Venv)
env₀ = leading_boundary(env₀, ψ₀, boundary_alg)

## Optimize

pepsopt_alg = PEPSOptimize(;
    boundary_alg=boundary_alg,
    optimizer=optimization_alg,
    gradient_alg=gradient_alg,
    reuse_env=reuse_env,
)
result = fixedpoint(ψ₀, H, pepsopt_alg, env₀)

@info "Finished $mode"

numfg = result.info
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
env₀ = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

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
env₀ = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

peps_cfun, peps_retract, peps_inner = peps_opt_costfunction(;
    boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(A, env), f, g, numfg, history = optimize(
    peps_cfun, (A0, env₀), optimization_alg; retract=peps_retract, inner=peps_inner
);

@info "Finished $mode"

@info "Energy: $E\t numfg: $numfg\t numiter: $(length(history[2]))"
