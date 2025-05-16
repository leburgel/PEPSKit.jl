# TODO: make pullback as a full matrix and check its condition number

"""
Some trials on the 'reverse' Jacobian linear problem for pulling-through fixed-point equations.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise

using Test
using LinearAlgebra
using TensorKit
using MPSKit
using PEPSKit
using KrylovKit
using Zygote
using VectorInterface

using PEPSKit: PEPSTensor, PartitionFunctionTensor
using MPSKitModels: classical_ising

# do the simple formulation for now
gauge = :center # switch gauge
style = :naive # switch inner product
# style = :regularized # modified inner product

# check 'reverse' Jacobian linear system for partition functions and PEPS norms
χ = 8
D = 3
d = 2

boundary_alg = PullingThrough(; tol=1e-12, verbosity=2, maxiter=500, gauge=gauge)

## Utility functions

# always working with rotation and Hermitian reflection symmetric local tensors
project_symmetric(x) = PEPSKit.herm_depth_inv(PEPSKit.rot_inv(x))

# modified inner product for pulling-through environmnent cotangents at a given fixed point
# TODO: will we get in trouble by storing the different components in a tuple and naively
# adding all the inners together?
function cotangent_inner(Xfp::MPSKit.MPSBondTensor)
    # TODO: figure out inverses; this seeds to be toggled in tandem with PEPSKit.precondition_environment_tangent
    Xprec = inv(Xfp)^2
    # Xprec = Xfp^2
    function dotf(
        (ΔA1, ΔX1, ΔN1)::Tuple{TA,TX,TN}, (ΔA2, ΔX2, ΔN2)::Tuple{TA,TX,TN}
    ) where {TA<:MPSKit.GenericMPSTensor,TX<:MPSKit.MPSBondTensor,TN<:Number}
        ΔA1ΔA2 = dot(ΔA1, PEPSKit.absorb_bond_matrices(ΔA2, Xprec, Xprec))
        ΔX1ΔX2 = tr(ΔX1' * ΔX2 * Xprec) / 4
        ΔN1ΔN2 = dot(ΔN1, ΔN2)
        return ΔA1ΔA2 + ΔX1ΔX2 + ΔN1ΔN2
    end
    return dotf
end

## Partition function case

@info "Testing 'reverse' Jacobian linear system for partition function contraction"
println()
println()

# start from random parititon function tensor
O = rand(ComplexF64, ℂ^D ⊗ ℂ^D ← ℂ^D ⊗ ℂ^D)

# make it nicely symmetric
O = project_symmetric(O)

# contract it using pulling through
state = InfinitePartitionFunction(O) # 'state'
env0 = PullingThroughEnv(state, ℂ^χ)
env, N, _ = leading_boundary(env0, state, boundary_alg)

# unpack the environment
A = env.A
X = env.X

# initialize the appropriate inner products
fp_dot = cotangent_inner(X)
input_inner = real ∘ dot
output_inner = style == :regularized ? real ∘ fp_dot : real ∘ dot

# construct the fixed-point equations in the appropriate form
function FP(state, A, X, N)
    return PEPSKit.pt_fixedpoint(
        Val(gauge), Val(style), InfiniteSquareNetwork(state), A, X, N
    )
end

# check how well they are satisfied
FPS = FP(state, A, X, N)
@info "Fixed-point equation residuals: $(norm.(FPS))"

# get the full pullback from Zygote
_, vjp = pullback(FP, state, A, X, N)

# restrict to environment pullback
function vjp_env_base(
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
# but make sure it works with the appropriate inner products
function vjp_env(ΔFPS::InnerProductVec{typeof(input_inner)})
    return InnerProductVec(vjp_env(ΔFPS.vec), output_inner)
end

# restrict to state pullback
vjp_state(ΔFP) = vjp(ΔFP)[1]

# get the pushforward
jvp_env = PEPSKit.generate_environment_pushforward(
    Val(gauge), Val(style), InfiniteSquareNetwork(state), A, X, N
)
# but make sure it works with the appropriate inner products
function jvp_envvv(∂env::InnerProductVec{typeof(output_inner)})
    return InnerProductVec(jvp_env(∂env.vec), input_inner)
end

# make the reverse Jacobian linear system

# initialize some random environment cotangents
ΔA = 1e-2 * PEPSKit.project_hermitian(randn(scalartype(A), space(A)))
ΔX = 1e-2 * PEPSKit.project_hermitian(randn(scalartype(A), space(X)))
ΔN = 1e-2 * complex(randn(Float64)) # set this one to zero for initial trials?
# toggle real diagonal projection of corner cotangent
ΔX = DiagonalTensorMap(ΔX) # can't make it manifestly real without mixing up scalartypes...
# make these into a right hand side
b = (ΔA, ΔX, ΔN)

# hack the inner products
b = InnerProductVec(b, output_inner)

solver_alg = LSMR(; tol=eps(), maxiter=5000, krylovdim=5000, verbosity=3)
ΔFPS, info = lssolve((vjp_env, jvp_envvv), b, solver_alg)
res = add(b, vjp_env(ΔFPS), -1) # should be small
@info "Normres after linsolve: $(norm(res))"
@info "Euclidean normres after linsolve: $(norm(res[]))"

# unpack solution
ΔFP2, ΔFP3, ΔFP4 = ΔFPS[]

# check hermiticity
@info "Checking hermiticity of the fixed-point cotangents obtained from the linear problem"
@show norm(ΔFP2 - PEPSKit.project_hermitian(ΔFP2))
@show norm(ΔFP3 - PEPSKit.project_hermitian(ΔFP3))
@show norm(ΔFP4 - real(ΔFP4))

# plop this into the state pullback and see what comes out
@info "Checking symmetry of the state cotangent obtained from the linear problem"
# TODO: add projectors to the state pullback
ΔT = -only(PEPSKit.unitcell(vjp_state(ΔFPS[])))
# and check the symmetries
ΔT̄ = PEPSKit._fit_spaces(PEPSKit.herm_depth(ΔT), ΔT)
ΔTr = PEPSKit._fit_spaces(PEPSKit.rotl90(ΔT), ΔT)
@show norm(ΔT - ΔT̄) / norm(ΔT) # should be small, and it is
@show norm(ΔT - ΔTr) / norm(ΔT) # so rotation invariance is not guaranteed at all, which makes sense
# TODO: should I just add more terms to make the fixed-point equations themselves symmetric? Or does this not make sense?
@show norm(ΔT - project_symmetric(ΔT)) / norm(ΔT) # should be small, but it's not at all...

# make the full environment Jacobian matrix and check its condition number
mat_tol = eps() / 100
orth = ModifiedGramSchmidt2()
iter = GKLIterator((vjp_env, jvp_envvv), b, orth)
fact = initialize(iter)
while normres(fact) > mat_tol
    expand!(iter, fact)
end
VJP_env = rayleighquotient(fact)
U, S, V = svd(VJP_env)

# check the sizes
ndof = dim(A) + 2
@info "Number of degrees of freedom: $ndof"
@info "Shape of full environment Jacobian matrix: $(size(VJP_env))"
@info "Condition number of full environment Jacobian matrix: $(cond(VJP_env))"
@info "Last 10 singular values: $(S[(end - 10):end])"

# TODO: more checks?

println()
println()

# ## PEPS case

# @info "Testing 'forward' Jacobian linear system for PEPS contraction"
# println()
# println()

# # start from random PEPS tensor
# T = rand(ComplexF64, ℂ^d ← ℂ^D ⊗ ℂ^D ⊗ (ℂ^D)' ⊗ (ℂ^D)')

# # make it nicely symmetric
# T = project_symmetric(T)

# # contract it using pulling through
# PEPS = InfinitePEPS(T)
# PEPSN = InfiniteSquareNetwork(PEPS)
# env0 = PullingThroughEnv(PEPS, ℂ^χ)
# env, N, _ = leading_boundary(env0, PEPS, boundary_alg)

# # unpack the environment
# A = env.A
# X = env.X

# # get the environment pushforward acting on (∂A, ∂X, ∂N)
# jvp_env = PEPSKit.generate_environment_pushforward(
#     Val(gauge), Val(style), PEPSN, env.A, env.X, N
# )

# # make the PEPS-tensor pushforward
# function generate_peps_pushforward(
#     A::MPSKit.GenericMPSTensor{S,3}, X::MPSKit.MPSBondTensor{S}
# ) where {S}
#     # duplicate this to have different bra and ket PEPS tensors...
#     function transfer_left(
#         A::MPSKit.GenericMPSTensor{S,3},
#         X::MPSKit.MPSBondTensor{S},
#         Tket::PEPSTensor{S},
#         Tbra::PEPSTensor{S},
#     ) where {S}
#         AU = PEPSKit.physical_flip(A)
#         PEPSKit.@autoopt @tensor A´[χ_SE D_E_above D_E_below; χ_NE] :=
#             A[χ_NNW D_N_above D_N_below; χ_NE] *
#             AU[χ_WSW D_W_above D_W_below; χ_WNW] *
#             AU[χ_SE D_S_above D_S_below; χ_SSW] *
#             X[χ_WNW; χ_NNW] *
#             X[χ_SSW; χ_WSW] *
#             Tket[d; D_N_above D_E_above D_S_above D_W_above] *
#             conj(Tbra[d; D_N_below D_E_below D_S_below D_W_below])
#         return PEPSKit.physical_flip(A´)
#     end
#     function jpv_T(∂T::PEPSTensor)
#         # FP2
#         ∂FP2 = transfer_left(A, X, ∂T, T) + transfer_left(A, X, T, ∂T)
#         # FP3
#         ∂FP3 = zeros(scalartype(A), space(X))
#         # FP4
#         ∂FP4 = zero(scalartype(A)) # TODO: figure out mixed scalartypes...

#         return (∂FP2, ∂FP3, ∂FP4)
#     end
#     return jpv_T
# end
# jvp_state = generate_peps_pushforward(A, X)

# # make the forward Jacobian linear system

# # initialize a random symmetric 'state' tangent
# dT = 1e-2 * project_symmetric(MPSKit.randomize!(copy(T)))
# # apply the state pushforward to get the right hand side of the linear system
# b = scale(jvp_state(dT), -1)
# x0 = (MPSKit.randomize!(copy(A)), MPSKit.randomize!(copy(X)), randn(scalartype(A)))
# solver_alg = GMRES(; tol=eps(), maxiter=1, krylovdim=5000, verbosity=2)
# x, info = reallinsolve(jvp_env, b, x0, solver_alg)

# # report:
# # completely stuck until iteration ~980
# # proper convergence between iterations ~1000 and ~1800, then residual flattens out again
# # residual falls below eps after ~2000 iterations

# # TODO: check the symmetries of the result
# dA, dX, dN = x

# # check edge hermiticity -> valid up to a phase?
# dĀ = PEPSKit.physical_flip(PEPSKit._conj(dA))
# @show ovlp = dot(dA, dĀ) / norm(dA) / norm(dĀ)
# phase = angle(ovlp)
# scale!(dA, exp(im * phase / 2))
# dĀ = PEPSKit.physical_flip(PEPSKit._conj(dA))
# @show norm(dA - dĀ) / norm(dA)

# # check corner hermiticity -> valid up to a phase?
# dX̄ = dX'
# @show ovlp = dot(dX, dX̄) / norm(dX) / norm(dX̄)
# phase = angle(ovlp)
# scale!(dX, exp(im * phase / 2))
# dX̄ = PEPSKit.physical_flip(PEPSKit._conj(dX))
# @show norm(dX - dX̄) / norm(dX)

# # check corner real-diagonality -> not valid at all...
# @show norm(imag(dX)) / norm(dX)
# @show norm(X - DiagonalTensorMap(dX)) / norm(dX)

# norm(b .- jvp_env(x)) # should be small

# # make the full environment Jacobian matrix and check its condition number
# mat_tol = eps()
# orth = ModifiedGramSchmidt2()
# iter = ArnoldiIterator(jvp_env, x0, orth)
# fact = initialize(iter)
# while normres(fact) > mat_tol
#     expand!(iter, fact)
# end
# JVP_env = rayleighquotient(fact)
# @show size(JVP_env)
# @show cond(JVP_env)
# U, S, V = svd(JVP_env)
# @show S[(end - 10):end]

# nothing

# # get full matrix representation of a function...
# using KrylovKit
# orth = ModifiedGramSchmidt2() # or any other scheme
# iter = ArnoldiIterator(f, v, orth)
# fact = initialize(iter)
# while normres(fact) > tol # e.g. tol = eps()
#     expand!(iter, fact)
# end
# A = rayleighquotient(fact) # matrix represention of `f`

# # TODO: apply this to the pullback in different gauges and styles, see what comes out