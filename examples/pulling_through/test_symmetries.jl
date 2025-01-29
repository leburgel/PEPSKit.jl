"""
Test properties and symmetries of a properly symmetrized pulling through environment.
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
using KrylovKit

#
# Contract
#

T = 2 / log(1 + sqrt(2))
χ = 12

O = classical_ising(; beta=1 / T)
P = O[1]

pt_state = InfinitePartitionFunction(P)
pt_envinit = PullingThroughEnv(pt_state, ℂ^χ)
pt_alg = PullingThrough(; tol=1e-10, maxiter=200, verbosity=3)

# run pure contraction, not 
pt_env, pt_λ, = PEPSKit.pulling_through_iterate(pt_envinit, pt_state, pt_alg)
pt_λ = abs(pt_λ)

#
# Symmetrize
#

symm_env, W = PEPSKit.symmetric_environment(pt_env)

#
# Test the symmetries
#

X = symm_env.X
A = symm_env.A
Up = symm_env.U

# check if we actually made N equal to W
ovlp = tr(A' * PEPSKit.apply_physical_operator(W, Up')) / (norm(A) * norm(W))
@show abs(ovlp)

## Check the fixed point equations

# check if X is normalized
@show tr(X^4)

# check if A is hermitian
Ā = PEPSKit.apply_physical_operator(PEPSKit._conj(A), Up')
@show norm(A - Ā)

# check the left fixed point of the transfer matrix
X2´ = MPSKit.transfer_left(X^2, A, A)
@show abs(tr(X2´' * X^2))

# check the eigenvalue equation
@tensor LHS[-1 -2; -3] :=
    A[1 3; -3] *
    PEPSKit.physical_flip(A)[5 4; 2] *
    PEPSKit.physical_flip(A)[-1 7; 6] *
    X[2; 1] *
    X[6; 5] *
    P[4 7; 3 -2]

@tensor RHS[-1 -2; -3] := PEPSKit.physical_flip(A)[1 -2; 2] * X[-1; 1] * X[2; -3]

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
@show norm(LHS - λ_out * RHS) # and the eigenvalue equation is satisfied

# check Bram's alternative with the adjoint to make it square
@tensor LHS2[-1 -2; -3] :=
    A[1 3; -3] *
    PEPSKit.physical_flip(A)[5 4; 2] *
    conj(A[6 7; -1]) *
    X[2; 1] *
    X[6; 5] *
    P[4 7; 3 -2]

@show norm(LHS2 - λ_out * RHS)
# works, so we should just pretend this is enough?
# don't need to include hermiticity in fixed-point equations?

# TODO: should we actually think of X as real for the fixed point equations, or not?
# TODO: add a PEPS test

nothing
