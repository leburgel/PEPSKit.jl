# combine pulling-through based optimization with proper spatial symmetries

using Revise

using TensorKit
using PEPSKit
using MPSKit
using KrylovKit
using OptimKit

include(joinpath(@__DIR__, "..", "spatial_symmetries", "spatial_toolbox.jl"))

# Setup
# -----

# model
J = 1.0
g = 3.0

# PEPS parameters
Dbond = 3
Denv = 20
symm_style = HReflectionRotation()
unitcell_style = Asymmetric() # flip south and west
optim_style = :manifest # use manifestly symmetric tensors
# optim_style = :manual # use manual symmetrization

# instantiate algorithm choices
alg_orth = Polar()
# alg_orth = QRpos() # unreasonably slow
dynamic_tols = true
# dynamic_tols = false
boundary_alg = PullingThrough(;
    tol=1e-10,
    maxiter=500,
    miniter=4,
    verbosity=2,
    dynamic_tols,
    alg_gauge=PEPSKit.Defaults.pt_alg_gauge(; dynamic_tols, alg_orth),
)
gradient_alg = PTLSSolver(;
    solver_alg=KrylovKit.LSMR(; maxiter=1_000, tol=1e-10, verbosity=2, krylovdim=1_000),
    gauge=:center,
    style=:naive,
)
ls_alg = ls_alg = BackTrackingLineSearch(; c₁=1e-4, maxiter=10, maxfg=10, maxstep=5.0)
optimizer_alg = LBFGS(
    10; acceptfirst=true, maxiter=100, gradtol=1e-4, verbosity=3, linesearch=ls_alg
)
reuse_env = true

# square lattice Heisenberg Hamiltonian
H = transverse_field_ising(lattice(unitcell_style); J, g)

# spaces
P = first(H.lattice)
Vpeps = ℂ^Dbond
Venv = ℂ^Denv

# Optimization
# ------------

if optim_style == :manual
    ## Manual

    # initialization
    A0 = TensorMap(randn, ComplexF64, P ← Vpeps ⊗ Vpeps ⊗ Vpeps ⊗ Vpeps)
    A0 = normalize(symmetrize(A0, symm_style))
    ψ0 = fill_peps(A0, unitcell_style)
    env0, = leading_boundary(PullingThroughEnv(ψ0, Venv), ψ0, boundary_alg)

    # optimization
    peps_cfun, peps_inner, peps_retract, peps_transport! = peps_opt_costfunction(
        H; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
    )

    (A, env), f, g = optimize(
        peps_cfun,
        (A0, env0),
        optimizer_alg;
        inner=peps_inner,
        retract=peps_retract,
        (transport!)=(peps_transport!),
    )

elseif optim_style == :manifest
    ## Manifest

    # initialization
    A_basis = find_symmetric_basis(P, Vpeps, symm_style)
    a0 = normalize(randn(length(A_basis)))
    ψ0 = fill_peps(vec2peps(a0, A_basis), unitcell_style)
    env0, = leading_boundary(PullingThroughEnv(ψ0, Venv), ψ0, boundary_alg)

    # optimization
    vector_cfun, vector_inner, vector_retract, vector_transport! = vector_opt_costfunction(
        H, P, Vpeps; boundary_alg, gradient_alg, reuse_env, unitcell_style, symm_style
    )

    (a, env), f, g, numfg, history = optimize(
        vector_cfun,
        (a0, env0),
        optimizer_alg;
        inner=vector_inner,
        retract=vector_retract,
        (transport!)=(vector_transport!),
    )
else
    throw(ArgumentError("Invalid optimization style!"))
end

nothing
