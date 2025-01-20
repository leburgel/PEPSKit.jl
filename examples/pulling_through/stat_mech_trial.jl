"""
Compare contraction methods for the 2D classical Ising partition function.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise

using LinearAlgebra
using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using PEPSKit
using QuadGK

#
# Setup
#

# proper error tracking for fair comparison
include("$(@__DIR__)/error_tracking.jl")

# compare to Onsager solution
function ising_free_energy(; beta=log(1 + sqrt(2)) / 2, J=1.0)
    K = beta * J
    k = 1 / sinh(2 * K)^2
    F = quadgk(
        theta -> log(cosh(2 * K)^2 + 1 / k * sqrt(1 + k^2 - 2 * k * cos(2 * theta))), 0, pi
    )[1]
    return -1 / beta * (log(2) / 2 + 1 / (2 * pi) * F)
end

#
# Contract
#

T = 2 / log(1 + sqrt(2))
χ = 12

O = classical_ising(; beta=1 / T)
P = O[1]

## CTMRG

ctm_errs = Float64[]
ctm_state = InfinitePartitionFunction(P)
ctm_envinit = CTMRGEnv(ctm_state, ℂ^χ)
ctm_finalize = ctm_error_tracker(ctm_errs, ctm_envinit)
ctm_alg = SimultaneousCTMRG(; tol=1e-12, maxiter=1000, verbosity=2, finalize=ctm_finalize)

ctm_env = leading_boundary(CTMRGEnv(ctm_state, ℂ^χ), ctm_state, ctm_alg)
ctm_λ = abs(PEPSKit.value(ctm_state, ctm_env))

## VUMPS

vumps_errs = Float64[]
vumps_state = O
vumps_envinit = InfiniteMPS(randn, ComplexF64, [ℂ^2], [ℂ^12])
vumps_finalize = vumps_error_tracker(vumps_errs, vumps_envinit)
vumps_alg = VUMPS(; tol=1e-12, maxiter=100, verbosity=2, finalize=vumps_finalize)

vumps_env, vumps_env_env, = leading_boundary(vumps_envinit, vumps_state, vumps_alg)
vumps_λ = abs(expectation_value(vumps_env, vumps_state, vumps_env_env))

## Pulling through

pt_errs = Float64[]
pt_state = InfinitePartitionFunction(P)
pt_envinit = PullingThroughEnv(pt_state, ℂ^χ)
pt_finalize = pt_error_tracker(pt_errs, pt_envinit)
pt_alg = PullingThrough(; tol=1e-12, maxiter=100, verbosity=2, finalize=pt_finalize)

pt_env, pt_λ, = leading_boundary(pt_envinit, pt_state, pt_alg)
pt_λ = abs(pt_λ)

#
# Verify
#

f_exact = ising_free_energy(; beta=1 / T)
@show abs(-log(ctm_λ) * T - f_exact)
@show abs(-log(vumps_λ) * T - f_exact)
@show abs(-log(pt_λ) * T - f_exact)

nothing
