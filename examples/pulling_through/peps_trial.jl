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

# proper error tracking for fair comparison
include("$(@__DIR__)/error_tracking.jl")

function load_tensor(λ::Float64, D::Int64)
    filename = "$(@__DIR__)/peps_ising_lambda_$(string(λ))_D_$(string(D)).txt"
    raw_data = parse.(ComplexF64, readlines(filename))

    # stored data has indexing convention W ⊗ N ⊗ E ⊗ S ← P
    t = TensorMap(raw_data, ℂ^D ⊗ (ℂ^D)' ⊗ (ℂ^D)' ⊗ ℂ^D ← (ℂ^2)')

    # transform to PEPSTensor indexing convention
    return permute(t, (5,), (2, 3, 4, 1))
end

#
# Contract
#

λ = 3.0
D = 3
χ = 12

t = load_tensor(λ, D)

## CTMRG

ctm_errs = Float64[]
ctm_state = InfinitePEPS(t)
ctm_envinit = CTMRGEnv(ctm_state, ℂ^χ)
ctm_finalize = ctm_error_tracker(ctm_errs, ctm_envinit)
ctm_alg = SimultaneousCTMRG(; tol=1e-12, maxiter=1000, verbosity=2, finalize=ctm_finalize)

ctm_env = leading_boundary(ctm_envinit, ctm_state, ctm_alg)

## VUMPS

vumps_errs = Float64[]
vumps_state = InfiniteTransferPEPS(InfinitePEPS(t), 1, 1)
vumps_env_init = initializeMPS(vumps_state, [ℂ^χ])
vumps_finalize = vumps_error_tracker(vumps_errs, vumps_env_init)
vumps_alg = VUMPS(; tol=1e-12, maxiter=200, verbosity=2, finalize=vumps_finalize)

vumps_env, vumps_env_env, = leading_boundary(vumps_env_init, vumps_state, vumps_alg)

## Pulling through

pt_errs = Float64[]
pt_state = InfinitePEPS(t)
pt_envinit = PullingThroughEnv(pt_state, ℂ^χ)
pt_finalize = pt_error_tracker(pt_errs, pt_envinit)
pt_alg = PullingThrough(; tol=1e-12, maxiter=200, verbosity=2, finalize=pt_finalize)

pt_env, = leading_boundary(pt_envinit, pt_state, pt_alg)

#
# Verify
#

lattice = InfiniteSquare(1, 1)
H = transverse_field_ising(ComplexF64, Trivial, lattice; J=1.0, g=λ)

e_expected = -3.194939968583713

# CTMRG expectation value is already implemented
ctm_e = expectation_value(ctm_state, H, ctm_env)

@show abs(ctm_e - e_expected)

# TODO: implement energy contractions for VUMPS and PullingThrough

nothing
