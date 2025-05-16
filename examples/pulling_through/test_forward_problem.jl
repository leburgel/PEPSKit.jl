"""
Some trials on the 'foward' Jacobian linear problem for pulling-through fixed-point equations.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise

using Test
using Random
using LinearAlgebra
using TensorKit
using MPSKit
using PEPSKit
using KrylovKit
using Zygote
using VectorInterface

using PEPSKit: PEPSTensor, PartitionFunctionTensor
using MPSKitModels: classical_ising

## Setup

Random.seed!(12345)

# do the simple formulation for now
gauge = :center # switch gauge
style = :naive # switch inner product for tangents and cotangents (this does not really work...)

# check 'forward' Jacobian linear system for partition functions and PEPS norms
χ = 12
D = 2
d = 2
beta = log(1 + sqrt(2)) / 2 # critical temperature
# beta = 0.43 # slightly above critical temperature

boundary_alg = PullingThrough(; tol=1e-12, verbosity=2, maxiter=500, gauge=gauge)

## Utility functions

# always working with rotation and Hermitian reflection symmetric local tensors
project_symmetric(x) = PEPSKit.herm_depth_inv(PEPSKit.rot_inv(x))

## Partition function case

@info "Testing 'forward' Jacobian linear system for partition function contraction"
println()
println()

# start from random parititon function tensor
O = classical_ising(ComplexF64, Trivial; beta)[1] # classical Ising partition function
# O = rand(ComplexF64, ℂ^D ⊗ ℂ^D ← ℂ^D ⊗ ℂ^D) # random mpo

# make it nicely symmetric
O = project_symmetric(O)

# contract it using pulling through
PF = InfinitePartitionFunction(O) # dummy parition function network
PFN = InfiniteSquareNetwork(PF)
env0 = PullingThroughEnv(PF, ℂ^χ)
env, N, ϵ = leading_boundary(env0, PF, boundary_alg)

# unpack the environment
A = env.A
X = env.X

# construct the fixed-point equations in the appropriate form
function FP(state, A, X, N)
    return PEPSKit.pt_fixedpoint(
        Val(gauge), Val(style), InfiniteSquareNetwork(state), A, X, N
    )
end
# check how well they are satisfied
FPS = FP(PF, A, X, N)
@info "Fixed-point equation residuals: $(norm.(FPS))"

# then construct dual-direction actions of the environment partial Jacobian

# get the environment pushforward acting on (∂A, ∂X, ∂N)
# this one is manually defined in the package itself
jvp_env = PEPSKit.generate_environment_pushforward(Val(gauge), Val(style), PFN, A, X, N)

# get the environment pullback acting on (ΔFP2, ΔFP3, ΔFP4)
# we can get this one automatically from just the fixed-point equations using Zygote
_, vjp = pullback(FP, PF, A, X, N)
# restrict to environment pullback
function vjp_env(
    (ΔFP2_, ΔFP3_, ΔFP4_)::Tuple{TA,TX,TN}
) where {TA<:MPSKit.GenericMPSTensor,TX<:MPSKit.MPSBondTensor,TN<:Number}
    if gauge == :center
        # toggle Hermitian projections
        ΔFP2_ = PEPSKit.project_hermitian(ΔFP2_)
        ΔFP3_ = PEPSKit.project_hermitian(ΔFP3_)
    end

    # apply pullback, isolate the environment part
    ΔA_, ΔX_, ΔN_ = vjp((ΔFP2_, ΔFP3_, ΔFP4_))[2:end]

    if gauge == :center
        # toggle Hermitian projections
        ΔA_ = PEPSKit.project_hermitian(ΔA_)
        ΔX_ = PEPSKit.project_hermitian(ΔX_)
        if X isa DiagonalTensorMap
            # toggle real diagonal projection of corner cotangent
            ΔX_ = DiagonalTensorMap(ΔX_) # cannot actually make the scalartype real without causing mixing issues later
        end
    end
    # TODO: project onto real and hermitian corners, see if this also works

    return (ΔA_, ΔX_, ΔN_)
end

# need the MPO-tensor pushforward
# this is not implemented in the package itself since we don't need it for reverse mode AD,
# but here we need it to get a sensible right hand side for our linear problem
function generate_partition_function_pushforward(A, X)
    function jpv_O(∂O::PartitionFunctionTensor)
        ∂O = project_symmetric(∂O)

        # FP2
        ∂FP2 = PEPSKit.fixed_point_2_transfer_naive(A, A, A, X, X, ∂O)
        # FP3
        ∂FP3 = zeros(scalartype(A), space(X))
        # FP4
        ∂FP4 = zero(scalartype(A)) # TODO: figure out mixed scalartypes...

        if gauge == :center
            # project tangents onto Hermitian components
            ∂FP2 = PEPSKit.project_hermitian(∂FP2)
            ∂FP3 = PEPSKit.project_hermitian(∂FP3)
        end

        return (∂FP2, ∂FP3, ∂FP4)
    end
    return jpv_O
end
jvp_state = generate_partition_function_pushforward(A, X)

# make the forward Jacobian linear system

# initialize a random symmetric 'state' tangent
dO = 1e-2 * project_symmetric(MPSKit.randomize!(copy(O)))

# apply the state pushforward to get the right hand side of the linear system
b = scale(jvp_state(dO), -1)

# solve the linear system
solver_alg = LSMR(; tol=eps(), maxiter=5000, krylovdim=5000, verbosity=3)
denv, info = reallssolve((jvp_env, vjp_env), b, solver_alg)
res = add(b, jvp_env(denv), -1) # should be small
@info "Normres after linsolve: $(norm.(res))"

# unpack solution
dA, dX, dN = denv

# check hermiticity
@info "Checking hermiticity of the environment tangents obtained from the linear problem"
@show norm(dA - PEPSKit.project_hermitian(dA))
@show norm(dX - PEPSKit.project_hermitian(dX))

# check corner real-diagonality -> not valid at all, but might be able to force it...
@info "Checking if the corner tangent obtained from the linear problem is real and diagonal"
@show norm(imag(dX)) / norm(dX)
@show norm(dX - DiagonalTensorMap(dX)) / norm(dX)

# make the full environment Jacobian matrix and check its condition number
mat_tol = eps()
orth = ModifiedGramSchmidt2()
iter = GKLIterator((jvp_env, vjp_env), b, orth)
fact = initialize(iter)
while normres(fact) > mat_tol
    expand!(iter, fact)
end
JVP_env = rayleighquotient(fact)
U, S, V = svd(JVP_env)

# check the sizes
ndof = dim(A) + dim(X) + 2
@info "Number of degrees of freedom: $ndof"
@info "Shape of full environment Jacobian matrix: $(size(JVP_env))"
@info "Condition number of full environment Jacobian matrix: $(cond(JVP_env))"
@info "Last 20 singular values: $(S[(end - 20):end])"

# check some more nullspace things: try an anti-Hermitian gauge transform
dy = randn(scalartype(A), space(X))
dy = (dy - dy') / 2
dAn = PEPSKit.absorb_left_bond_matrix(A, dy) - PEPSKit.absorb_right_bond_matrix(A, dy)
dXn = dy * X - X * dy

out = jvp_env((dAn, dXn, zero(N)))
@info "Image of infinitessimal gauge transform under environment Jacobian: norm.(jvp_env(denv)) = $(norm.(out))"

# find the gauge transformation that makes the corner tangent diagonal and real
@info "Using this to diagonalize the corner tangent"
dy_mat = zeros(scalartype(dX), χ, χ)
X_mat = X[]
dX_mat = dX[]
for i in 1:χ
    for j in 1:χ
        i == j && continue
        dy_mat[i, j] = -dX_mat[i, j] / (X_mat[j, j] - X_mat[i, i])
    end
end
dy = TensorMap(dy_mat, ℂ^χ ← ℂ^χ)
# check that this is antihermitian
@info "Checking that the diagonalizing gauge transformation is antihermitian: norm(dy + dy') = $(norm(dy + dy'))"

# transform the corner and edge tangents, and check if this is also a solution to the linear
dA´ = dA + PEPSKit.absorb_left_bond_matrix(A, dy) - PEPSKit.absorb_right_bond_matrix(A, dy)
dX´ = dX + dy * X - X * dy

# check that the corner tangent is indeed diagonal now
offdiag_res = norm(dX´ - DiagonalTensorMap(dX´))
imag_res = norm(imag(dX´))
@info "Off-diagonal component of the transformed corner tangent: norm(dX´ - diag(dX´)) = $offdiag_res"
@info "Imaginary component of the transformed corner tangent: norm(imag(dX´)) = $imag_res"
dX´ = DiagonalTensorMap(real(dX´))
dA´ = PEPSKit.project_hermitian(dA´) # reproject?

# check that this is still a solution to the linear system
@info "Normres after gauge transformation: $(norm.(b .- jvp_env((dA´, dX´, dN))))"
# so, does this mean we can always project the corner cotangents to be real and diagonal?

println()
println()

## PEPS case

@info "Testing 'forward' Jacobian linear system for PEPS contraction"
println()
println()

# start from random PEPS tensor
T = rand(ComplexF64, ℂ^d ← ℂ^D ⊗ ℂ^D ⊗ (ℂ^D)' ⊗ (ℂ^D)')

# make it nicely symmetric
T = project_symmetric(T)

# contract it using pulling through
PEPS = InfinitePEPS(T)
PEPSN = InfiniteSquareNetwork(PEPS)
env0 = PullingThroughEnv(PEPS, ℂ^χ)
env, N, _ = leading_boundary(env0, PEPS, boundary_alg)

# unpack the environment
A = env.A
X = env.X

# check how well they are satisfied
FPS = FP(PEPS, A, X, N)
@info "Fixed-point equation residuals: $(norm.(FPS))"

# then construct dual-direction actions of the environment partial Jacobian

# get the environment pushforward acting on (∂A, ∂X, ∂N)
# this one is manually defined in the package itself
jvp_env = PEPSKit.generate_environment_pushforward(Val(gauge), Val(style), PEPSN, A, X, N)

# get the environment pullback acting on (ΔFP2, ΔFP3, ΔFP4)
# we can get this one automatically from just the fixed-point equations using Zygote
_, vjp = pullback(FP, PEPS, A, X, N)
# restrict to environment pullback
function vjp_env(
    (ΔFP2_, ΔFP3_, ΔFP4_)::Tuple{TA,TX,TN}
) where {TA<:MPSKit.GenericMPSTensor,TX<:MPSKit.MPSBondTensor,TN<:Number}
    if gauge == :center
        # toggle Hermitian projections
        ΔFP2_ = PEPSKit.project_hermitian(ΔFP2_)
        ΔFP3_ = PEPSKit.project_hermitian(ΔFP3_)
    end

    # apply pullback, isolate the environment part
    ΔA_, ΔX_, ΔN_ = vjp((ΔFP2_, ΔFP3_, ΔFP4_))[2:end]

    if gauge == :center
        # toggle Hermitian projections
        ΔA_ = PEPSKit.project_hermitian(ΔA_)
        ΔX_ = PEPSKit.project_hermitian(ΔX_)
        if X isa DiagonalTensorMap
            # toggle real diagonal projection of corner cotangent
            ΔX_ = DiagonalTensorMap(ΔX_) # cannot actually make the scalartype real without causing mixing issues later
        end
    end
    # TODO: project onto real and hermitian corners, see if this also works

    return (ΔA_, ΔX_, ΔN_)
end

# make the PEPS-tensor pushforward
function generate_peps_pushforward(
    A::MPSKit.GenericMPSTensor{S,3}, X::MPSKit.MPSBondTensor{S}
) where {S}
    # duplicate this to have different bra and ket PEPS tensors...
    function transfer_left(
        A::MPSKit.GenericMPSTensor{S,3},
        X::MPSKit.MPSBondTensor{S},
        Tket::PEPSTensor{S},
        Tbra::PEPSTensor{S},
    ) where {S}
        AU = PEPSKit.physical_flip(A)
        PEPSKit.@autoopt @tensor A´[χ_SE D_E_above D_E_below; χ_NE] :=
            A[χ_NNW D_N_above D_N_below; χ_NE] *
            AU[χ_WSW D_W_above D_W_below; χ_WNW] *
            AU[χ_SE D_S_above D_S_below; χ_SSW] *
            X[χ_WNW; χ_NNW] *
            X[χ_SSW; χ_WSW] *
            Tket[d; D_N_above D_E_above D_S_above D_W_above] *
            conj(Tbra[d; D_N_below D_E_below D_S_below D_W_below])
        return PEPSKit.physical_flip(A´)
    end
    function jpv_T(∂T::PEPSTensor)
        # FP2
        ∂FP2 = transfer_left(A, X, ∂T, T) + transfer_left(A, X, T, ∂T)
        # FP3
        ∂FP3 = zeros(scalartype(A), space(X))
        # FP4
        ∂FP4 = zero(scalartype(A)) # TODO: figure out mixed scalartypes...

        if gauge == :center
            # toggle Hermitian projections
            ∂FP2 = PEPSKit.project_hermitian(∂FP2)
            ∂FP3 = PEPSKit.project_hermitian(∂FP3)
        end

        return (∂FP2, ∂FP3, ∂FP4)
    end
    return jpv_T
end
jvp_state = generate_peps_pushforward(A, X)

# make the forward Jacobian linear system

# initialize a random symmetric 'state' tangent
dT = 1e-2 * project_symmetric(MPSKit.randomize!(copy(T)))

# apply the state pushforward to get the right hand side of the linear system
b = scale(jvp_state(dT), -1)

# initialize random environment tangents as a starting guess, and make it nicely symmetric
x0 = (
    PEPSKit.project_hermitian(MPSKit.randomize!(copy(FPS[1]))),
    PEPSKit.project_hermitian(MPSKit.randomize!(copy(FPS[2]))),
    randn(scalartype(A)),
)

# solve the linear system
solver_alg = LSMR(; tol=eps(), maxiter=5000, krylovdim=5000, verbosity=3)
x, info = reallssolve((jvp_env, vjp_env), b, solver_alg)
res = add(b, jvp_env(x), -1) # should be small
@info "Normres after linsolve: $(norm.(res))"

# unpack solution
dA, dX, dN = x

# check hermiticity
@info "Checking hermiticity of the environment tangents obtained from the linear problem"
@show norm(dA - PEPSKit.project_hermitian(dA))
@show norm(dX - PEPSKit.project_hermitian(dX))

# check corner real-diagonality -> not valid at all, but might be able to force it...
@info "Checking if the corner tangent obtained from the linear problem is real and diagonal"
@show norm(imag(dX)) / norm(dX)
@show norm(dX - DiagonalTensorMap(dX)) / norm(dX)

# make the full environment Jacobian matrix and check its condition number
mat_tol = 1e-14
orth = ModifiedGramSchmidt2()
iter = GKLIterator((jvp_env, vjp_env), x0, orth)
fact = initialize(iter)
while normres(fact) > mat_tol
    expand!(iter, fact)
end
JVP_env = rayleighquotient(fact)
U, S, V = svd(JVP_env)

# check the sizes
ndof = dim(A) + 2
@info "Number of degrees of freedom: $ndof"
@info "Shape of full environment Jacobian matrix: $(size(JVP_env))"
@info "Condition number of full environment Jacobian matrix: $(cond(JVP_env))"
@info "Last 10 singular values: $(S[(end - 20):end])"

# check some more nullspace things: try an anti-Hermitian gauge transform
dy = randn(scalartype(A), space(X))
dy = (dy - dy') / 2
dAn = PEPSKit.absorb_left_bond_matrix(A, dy) - PEPSKit.absorb_right_bond_matrix(A, dy)
dXn = dy * X - X * dy

out = jvp_env((dAn, dXn, zero(N)))
@info "Image of infinitessimal gauge transform under environment Jacobian: norm.(jvp_env(denv)) = $(norm.(out))"

@info "Using this to diagonalize the corner tangent"

# find the gauge transformation that makes the corner tangent diagonal and real
@info "Using this to diagonalize the corner tangent"
dy_mat = zeros(scalartype(dX), χ, χ)
X_mat = X[]
dX_mat = dX[]
for i in 1:χ
    for j in 1:χ
        i == j && continue
        dy_mat[i, j] = -dX_mat[i, j] / (X_mat[j, j] - X_mat[i, i])
    end
end
dy = TensorMap(dy_mat, ℂ^χ ← ℂ^χ)
# check that this is antihermitian
@info "Checking that the diagonalizing gauge transformation is antihermitian: norm(dy + dy') = $(norm(dy + dy'))"

# transform the corner and edge tangents, and check if this is also a solution to the linear
dA´ = dA + PEPSKit.absorb_left_bond_matrix(A, dy) - PEPSKit.absorb_right_bond_matrix(A, dy)
dX´ = dX + dy * X - X * dy

# check that the corner tangent is indeed diagonal now
offdiag_res = norm(dX´ - DiagonalTensorMap(dX´))
imag_res = norm(imag(dX´))
@info "Off-diagonal component of the transformed corner tangent: norm(dX´ - diag(dX´)) = $offdiag_res"
@info "Imaginary component of the transformed corner tangent: norm(imag(dX´)) = $imag_res"
dX´ = DiagonalTensorMap(real(dX´))
dA´ = PEPSKit.project_hermitian(dA´) # reproject?

# check that this is still a solution to the linear system
@info "Normres after gauge transformation: $(norm.(b .- jvp_env((dA´, dX´, dN))))"
# so, does this mean we can always project the corner cotangents to be real and diagonal?

nothing
