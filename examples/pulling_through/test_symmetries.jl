"""
Test properties and symmetries of a properly symmetrized pulling through environment.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise

using LinearAlgebra
using TensorKit
using MPSKit
using MPSKitModels
using PEPSKit
using KrylovKit

#
# Partition function
#

@info "Testing pulling through fixed-point equations for partition function contraction"

χ = 12

O = classical_ising(; beta=log(1 + sqrt(2)) / 2)
P = O[1]

pt_state = InfinitePartitionFunction(P)
pt_envinit = PullingThroughEnv(pt_state, ℂ^χ)
pt_alg = PullingThrough(; tol=1e-10, maxiter=200, verbosity=3)

# run pure contraction, without the symmetrization step
pt_env, pt_λ, = PEPSKit.pulling_through_iterate(
    pt_envinit, InfiniteSquareNetwork(pt_state), pt_alg
)
pt_λ = abs(pt_λ)

## Symmetrize in center gauge

gauge = :center
symm_env, Up, W, λ_transfer = PEPSKit.center_gauge_environment(pt_env)

## Test the fixed-point equations

X = symm_env.X
A = symm_env.A

# check if we actually made N equal to W
ovlp = tr(A' * PEPSKit.apply_physical_unitary(W, Up')) / (norm(A) * norm(W))
@show abs(ovlp)

## Check the fixed point equations

for style in [:naive, :regularized]
    @info "Style: $style"

    # FP1
    @show norm(PEPSKit.fixed_point_1(Val(gauge), Val(style), A, X))

    # FP2
    @show norm(PEPSKit.fixed_point_2(Val(gauge), Val(style), A, X, pt_λ, P))

    # FP3
    @show norm(PEPSKit.fixed_point_3(Val(gauge), Val(style), A, X))

    # FP4
    @show norm(PEPSKit.fixed_point_4(Val(gauge), Val(style), X))
end

@info "Checking local contraction"

λ_out = @tensor A[1 3; 8] *
    PEPSKit.physical_flip(A)[5 4; 2] *
    PEPSKit.physical_flip(A)[12 7; 6] *
    A[9 10; 11] *
    X[2; 1] *
    X[6; 5] *
    X[11; 12] *
    X[8; 9] *
    P[4 7; 3 10]

@show abs(λ_out - pt_λ) # can get eigenvalue from local contraction

## Symmetrize in left gauge

gauge = :left
symm_env, Up, W, λ_transfer = PEPSKit.left_gauge_environment(pt_env)

## Check the fixed point equations

X = symm_env.X
A = symm_env.A

# check if we actually made N equal to W
ovlp = tr(A' * PEPSKit.apply_physical_unitary(W, Up')) / (norm(A) * norm(W))
@show abs(ovlp)

## Check the fixed point equations

for style in [:naive, :regularized]
    @info "Style: $style"

    # FP1
    @show norm(PEPSKit.fixed_point_1(Val(gauge), Val(style), A, X))

    # FP2
    @show norm(PEPSKit.fixed_point_2(Val(gauge), Val(style), A, X, pt_λ, P))

    # FP3
    @show norm(PEPSKit.fixed_point_3(Val(gauge), Val(style), A, X))

    # FP4
    @show norm(PEPSKit.fixed_point_4(Val(gauge), Val(style), X))
end

@info "Checking local contraction"

λ_out = @tensor A[1 3; 8] *
    PEPSKit.physical_flip(A)[5 4; 2] *
    PEPSKit.physical_flip(A)[12 7; 6] *
    A[9 10; 11] *
    X[2; 1] *
    X[6; 5] *
    X[11; 12] *
    X[8; 9] *
    P[4 7; 3 10]

@show abs(λ_out - pt_λ) # can get eigenvalue from local contraction

# TODO: look into making fixed-point equation system square

nothing
