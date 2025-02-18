# incorporate spatial symmetries in PEPS optimization: example using Heisenberg model

using Revise

using LinearAlgebra
using KrylovKit
using TensorKit
using PEPSKit
import MPSKitModels:
    MPSKitModels, bose_hubbard_model, a_plusmin, a_minplus, a_number, contract_onesite
using Zygote
using ChainRulesCore
using OptimKit

using PEPSKit: PEPSTensor

# Part O: Setup
# -------------

include("spatial_toolbox.jl")
include("u1_toolbox.jl")
include("space_shifting.jl")

# TODO: add this to PEPSKit itself...
function MPSKitModels.bose_hubbard_model(
    elt::Type{<:Number},
    symmetry::Type{<:Sector},
    lattice::InfiniteSquare;
    cutoff::Integer=5,
    t=1.0,
    U=1.0,
    mu=0.0,
    n::Integer=0,
)
    @assert n == 0 "Currently no support for imposing a fixed particle number"
    hopping_term =
        a_plusmin(elt, symmetry; cutoff=cutoff) + a_minplus(elt, symmetry; cutoff=cutoff)
    N = a_number(elt, symmetry; cutoff=cutoff)
    interaction_term = contract_onesite(N, N - id(domain(N)))

    spaces = fill(space(N, 1), (lattice.Nrows, lattice.Ncols))

    H = LocalOperator(
        spaces,
        (neighbor => -t * hopping_term for neighbor in nearest_neighbours(lattice))...,
        ((idx,) => U / 2 * interaction_term - mu * N for idx in vertices(lattice))...,
    )

    return H
end

# spaces
symmetry = U1Irrep
Vpeps = U1Space(0 => 2, 1//2 => 1, -1//2 => 1) # TODO: play around with this
Venv = U1Space(0 => 4, 1//2 => 2, -1//2 => 2, 1 => 2, -1 => 2) # TODO: seed with a dynamic pass...

# uniform auxiliary physical space, for uniform half-filling
Paux = U1Space(-1//2 => 1)

# parameters
t = 1.0
U = 5.0
cutoff = 1

boundary_alg = SimultaneousCTMRG(;
    trscheme=FixedSpaceTruncation(), tol=1e-10, miniter=3, maxiter=400, verbosity=2
)
gradient_alg = LinSolver(;
    solver=GMRES(; tol=1e-6, maxiter=10, verbosity=2), iterscheme=:diffgauge
) # :diffgauge necessary for :sequential CTMRG scheme
optimization_alg = LBFGS(; gradtol=1e-4, verbosity=3)
reuse_env = true

# choose symmetrization style: needs to be U1-compatible!
# symm_style = None()
# symm_style = Rotation()
symm_style = U1HReflectionRotation() # so we need charge conjugation to make things match!

# choose unit cell style: still uniform so can just use the same filling styles as before
# unitcell_style = Asymmetric()
unitcell_style = Symmetric()

# square lattice Bose-Hubbard Hamiltonian
bose_hubbard_ham(::Asymmetric, args...; kwargs...) = bose_hubbard_model(args...; kwargs...)
function bose_hubbard_ham(::Symmetric, args...; kwargs...)
    return repeat(bose_hubbard_ham(Asymmetric(), args...; kwargs...), 2, 2)
end

# shift Hamiltonian and record shifted physical spaces
H1 = bose_hubbard_ham(unitcell_style, symmetry, InfiniteSquare(); t, U, cutoff)
H, Pspaces = shift_physical_spaces(H1, fill(Paux, size(H1.lattice)))
P = first(Pspaces) # uniform physical space

# Part I: manually imposing symmetries in gradient computation and retraction
# ---------------------------------------------------------------------------

## Initialization

# ititialize state and manually symmetrize
A0 = TensorMap(randn, ComplexF64, P ← Vpeps ⊗ Vpeps ⊗ Vpeps ⊗ Vpeps)
A0 = symmetrize(A0, symm_style)
ψ₀ = fill_peps(A0, unitcell_style)
env₀, = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

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
env₀, = leading_boundary(CTMRGEnv(ψ₀, Venv), ψ₀, boundary_alg)

## Optimization

vector_cfun, vector_retract, vector_inner = vector_opt_costfunction(
    P, Vpeps; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
)

(a, env), f, g, numfg, history = optimize(
    vector_cfun, (a₀, env₀), optimization_alg; retract=vector_retract, inner=vector_inner
);

nothing
