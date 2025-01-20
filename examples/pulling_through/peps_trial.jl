"""
Compare contraction methods for an optimized PEPS ground state of the 2+1D transverse field
Ising model.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise

using LinearAlgebra
using TensorKit
using MPSKit
using MPSKitModels
using PEPSKit

#
# Setup
#

function load_tensor(λ::Float64, D::Int64)
    filename = "$(@__DIR__)/peps_ising_lambda_$(string(λ))_D_$(string(D)).txt"
    raw_data = parse.(ComplexF64, readlines(filename))

    # stored data has indexing convention W ⊗ N ⊗ E ⊗ S ← P
    t = TensorMap(raw_data, ℂ^D ⊗ (ℂ^D)' ⊗ (ℂ^D)' ⊗ ℂ^D ← (ℂ^2)')

    # transform to PEPSTensor indexing convention
    return permute(t, (5,), (2, 3, 4, 1))
end

λ = 3.0
D = 3
χ = 12

t = load_tensor(λ, D)

## CTMRG

ctm_errs = []
function ctm_finalize(iter, η, env, state)
    push!(ctm_errs, η)
    return env
end
ctm_alg = SimultaneousCTMRG(; tol=1e-12, maxiter=1000, verbosity=2, finalize=ctm_finalize)
ctm_state = InfinitePEPS(t)
ctm_env = leading_boundary(CTMRGEnv(ctm_state, ℂ^χ), ctm_state, ctm_alg)

## VUMPS

vumps_errs = []
function vumps_finalize(iter, state, op, envs)
    η = MPSKit.calc_galerkin(state, envs) # terrible...
    push!(vumps_errs, η)
    return state, envs
end
vumps_alg = VUMPS(; tol=1e-12, maxiter=100, verbosity=2, finalize=vumps_finalize)
vumps_state = InfiniteTransferPEPS(InfinitePEPS(t), 1, 1)
vumps_env_init = initializeMPS(vumps_state, [ℂ^χ])
vumps_env, = leading_boundary(vumps_env_init, vumps_state, vumps_alg)

## Pulling through

pt_errs = []
function pt_finalize(iter, η, env, state)
    push!(pt_errs, η)
    return env
end
pt_alg = PullingThrough(; tol=1e-12, maxiter=100, verbosity=2, finalize=pt_finalize)
pt_state = InfinitePEPS(t)
pt_env, = leading_boundary(PullingThroughEnv(pt_state, ℂ^χ), pt_state, pt_alg)

#
# Verify
#

# TODO: test on transverse field Ising model

lattice = InfiniteSquare(1, 1)
H = transverse_field_ising(ComplexF64, Trivial, lattice; J=1.0, g=λ / 4)

ctm_e = expectation_value(ctm_state, H, ctm_env)
# supposed to be e = -3.194939968583713?

nothing
