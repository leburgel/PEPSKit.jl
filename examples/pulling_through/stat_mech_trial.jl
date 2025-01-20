"""
Compare contraction methods for the 2D classical Ising partition function.
"""

import Pkg
Pkg.activate(".")

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

function ising_free_energy(; beta=log(1 + sqrt(2)) / 2, J=1.0)
    K = beta * J
    k = 1 / sinh(2 * K)^2
    F = quadgk(
        theta -> log(cosh(2 * K)^2 + 1 / k * sqrt(1 + k^2 - 2 * k * cos(2 * theta))), 0, pi
    )[1]
    return -1 / beta * (log(2) / 2 + 1 / (2 * pi) * F)
end

T = 2 / log(1 + sqrt(2))
χ = 12

O = classical_ising(; beta=1 / T)
P = O[1]

#
# Contract
#

## CTMRG

ctm_alg = SimultaneousCTMRG(; tol=1e-12, maxiter=1000, verbosity=3)
ctm_state = InfinitePartitionFunction(P)
ctm_env = leading_boundary(CTMRGEnv(ctm_state, ℂ^χ), ctm_state, ctm_alg)
ctm_λ = abs(PEPSKit.value(ctm_state, ctm_env))

## VUMPS

vumps_alg = VUMPS(; tol=1e-12, maxiter=100, verbosity=3)
vumps_state = O
vumps_env, vumps_env_env, = leading_boundary(
    InfiniteMPS(randn, ComplexF64, [ℂ^2], [ℂ^12]), vumps_state, vumps_alg,
)
vumps_λ = abs(expectation_value(vumps_env, vumps_state, vumps_env_env))

## Pulling through

pt_alg = PullingThrough(; tol=1e-12, maxiter=100, verbosity=3)
pt_state = InfinitePartitionFunction(P)
pt_env, pt_λ, = leading_boundary(PullingThroughEnv(pt_state, ℂ^χ), pt_state, pt_alg)
pt_λ = abs(pt_λ)

#
# Verify
#

f_exact = ising_free_energy(; beta=1 / T)
@show abs(-log(ctm_λ) * T - f_exact)
@show abs(-log(vumps_λ) * T - f_exact)
@show abs(-log(pt_λ) * T - f_exact)

nothing
