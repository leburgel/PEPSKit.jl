
function _corner_type(A::MPSKit.GenericMPSTensor{S,N}) where {S,N}
    return TensorKit.tensormaptype(S, 1, 1, scalartype(A))
end

#
# Iteration environment
#

"""
    struct PullingThroughEnv{C,T}

Pulling through environment containing north and west edge tensors in different gauges, and
their corresponding left and right gauging bond tensors.

This is used in an routine to contract an infinite translation-invariant square network
built from a rank-4-like local tensors `T`, by iteratively solving the following fixed-point
equation:

```
         --<--N--<--                    --<--
        |     |                        |
        v     v                        v
        |     |                        |
        W--<--T--<--  = λ ⋅            W--<---
        |     |                        |
        v     v                        v
        |     |                        |
   --<--                    --<--N--<--
                                 |
                                 v
                                 |
```

The relation between the different tensors are:

```
--<--NL--<-- = --<--LN--<--N--<--LN^{-1}--<--
     |                     |
     v                     v
     |                     |
```

```
               |
               v
               |
               RW
     |         |
     v         v
     |         |
     WR--<-- = W--<--
     |         |
     v         v
     |         |
               RW^{-1}
               |
               v
               |
```
"""
struct PullingThroughEnv{C<:MPSKit.MPSBondTensor,T<:MPSKit.GenericMPSTensor}
    N::T
    W::T
    NL::Union{Nothing,T}
    WR::Union{Nothing,T}
    LN::Union{Nothing,C}
    RW::Union{Nothing,C}
    function PullingThroughEnv(
        N::T,
        W::T,
        NL::Union{Nothing,T},
        WR::Union{Nothing,T},
        LN::Union{Nothing,C},
        RW::Union{Nothing,C},
    ) where {C,T}
        Cout = _corner_type(N)
        return new{Cout,T}(N, W, NL, WR, LN, RW)
    end
end

function PullingThroughEnv(N, W; NL=nothing, WR=nothing, LN=nothing, RW=nothing)
    return PullingThroughEnv(N, W, NL, WR, LN, RW)
end

## Constructors

# symmetric pulling through, so we only use a single virtual space

# allow constructing environments for implicitly defined contractible networks
function PullingThroughEnv(state::Union{InfinitePartitionFunction,InfinitePEPS}, args...)
    return PullingThroughEnv(InfiniteSquareNetwork(state), args...)
end
function PullingThroughEnv(
    f, T, state::Union{InfinitePartitionFunction,InfinitePEPS}, args...
)
    return PullingThroughEnv(f, T, InfiniteSquareNetwork(state), args...)
end

# fill in default eltype and data initializer
function PullingThroughEnv(network::InfiniteSquareNetwork, chi::ElementarySpaceLike)
    return PullingThroughEnv(randn, ComplexF64, network, chi)
end

# actual constructor
function PullingThroughEnv(
    f, ::Type{T}, network::InfiniteSquareNetwork, chi::ElementarySpaceLike
) where {T}
    chi = _to_space(chi)
    P = network[1, 1] # hardcoded to single-site unit cell for now
    N = TensorMap(f, T, chi ⊗ _elementwise_dual(north_virtualspace(P)) ← chi)
    W = TensorMap(f, T, chi ⊗ _elementwise_dual(west_virtualspace(P)) ← chi)

    return PullingThroughEnv(N, W)
end

## Gauging

function gauge_north(env::PullingThroughEnv, alg_gauge)
    N = env.N
    LN0 = if isnothing(env.LN)
        isomorphism(storagetype(N), space(N, 1), space(N, 1))
    else
        env.LN
    end
    leftorth_alg = MPSKit.LeftCanonical(;
        tol=alg_gauge.tol,
        maxiter=alg_gauge.maxiter,
        verbosity=alg_gauge.verbosity,
        alg_orth=alg_gauge.alg_orth,
    )
    AL = MPSKit.PeriodicArray([copy(N)])
    C = MPSKit.PeriodicArray([copy(LN0)])

    MPSKit.uniform_leftorth!((AL, C), [N], LN0, leftorth_alg)

    @reset env.NL = only(AL)
    @reset env.LN = only(C)

    return env
end

_right_canonical_orth_alg(alg_orth) = alg_orth
_right_canonical_orth_alg(::QRpos) = LQpos() # this is very annoying...

function gauge_west(env::PullingThroughEnv, alg_gauge)
    W = env.W
    RW0 = if isnothing(env.RW)
        isomorphism(storagetype(W), space(W, 1), space(W, 1))
    else
        env.RW
    end

    rightorth_alg = MPSKit.RightCanonical(;
        tol=alg_gauge.tol,
        maxiter=alg_gauge.maxiter,
        verbosity=alg_gauge.verbosity,
        alg_orth=_right_canonical_orth_alg(alg_gauge.alg_orth),
    )
    AR = MPSKit.PeriodicArray([copy(W)])
    C = MPSKit.PeriodicArray([copy(RW0)])

    MPSKit.uniform_rightorth!((AR, C), [W], RW0, rightorth_alg)

    @reset env.WR = only(AR)
    @reset env.RW = only(C)

    return env
end

#
# Symmetrization contractions
#

import TensorKitManifolds as TKM

const SquareTensorMap{S,N} = AbstractTensorMap{<:Any,S,N,N}

function normalize_mps(A::MPSKit.GenericMPSTensor; tol=1e-14)
    init = MPSKit.randomize!(similar(A, space(A, 1), space(A, 1)))
    λ, = MPSKit.fixedpoint(flip(MPSKit.TransferMatrix(A, A)), init, :LM; tol=tol)
    return A / sqrt(abs(λ))
end

"""
Initialize an isometry between the physical spaces of two MPS tensors.
"""
function physical_isometry(
    N::MPSKit.GenericMPSTensor{S,N₁}, W::MPSKit.GenericMPSTensor{S,N₁}
) where {S,N₁}
    PN = prod(2:N₁) do i
        return space(N, i)
    end
    PW = prod(2:N₁) do i
        return space(W, i)
    end
    return isomorphism(storagetype(N), PW, PN)
end

"""
    transfer_left(Q, N, W, U)

Apply a (generalized) transfer matrix to the left.

```
 ┌←N←
 ↓ ↓
 Q U
 ↓ ↓
 └→̄W→
```
"""
@generated function transfer_left(
    Q::SquareTensorMap{S,1},
    N::MPSKit.GenericMPSTensor{S,N₁},
    W::MPSKit.GenericMPSTensor{S,N₁},
    U::SquareTensorMap{S,N₂},
) where {S,N₁,N₂}
    # TODO: assert that N₁ and N₂ are consistent?
    Q_out_e = tensorexpr(:Q_out, -1, -2)
    Q_e = tensorexpr(:Q, 1, 2 + 2 * N₂)
    N_e = tensorexpr(:N, (2 + 2 * N₂, ((1:N₂) .+ N₁)...), -2)
    W_e = tensorexpr(:W, (1, ((1:N₂) .+ 1)...), -1)
    U_e = tensorexpr(:U, (1:N₂) .+ 1, (1:N₂) .+ N₁)
    return macroexpand(
        @__MODULE__, :(return @tensor $Q_out_e := $Q_e * $N_e * conj($W_e) * $U_e)
    )
end

# generate single-argument transfer function based on current environment
function gen_transfer_left(N, W, U)
    tf(Q) = transfer_left(Q, N, W, U)
    return tf
end

"""
    physical_env(N, W, Q)

Contract the virtual indices of an MPS transfer matrix to give a physical reduced density
matrix.

```
 ┌←N←┐
 ↓ ↓ ↑
 Q   Q†
 ↓ ↓ ↑
 └→̄W→┙
```
"""
@generated function physical_env(
    N::MPSKit.GenericMPSTensor{S,N₁},
    W::MPSKit.GenericMPSTensor{S,N₁},
    Q::SquareTensorMap{S,1},
) where {S,N₁}
    N₂ = N₁ - 1
    U_e = tensorexpr(:U, -(1:N₂), -((1:N₂) .+ N₂))
    N_e = tensorexpr(:N, (2, (-((1:N₂) .+ N₂))...), 3)
    W_e = tensorexpr(:W, (1, (-(1:N₂))...), 4)
    Q_e = tensorexpr(:Q, 1, 2)
    Q̄_e = tensorexpr(:Q, 4, 3)
    return macroexpand(
        @__MODULE__, :(return @tensor $U_e := $Q_e * $N_e * conj($W_e) * conj($Q̄_e))
    )
end

"""
Trivial physical flipper. TODO: try to not make an implicit choice at some point...
"""
function physical_flipper(N::MPSKit.GenericMPSTensor{S,N₁}) where {S,N₁}
    PN = prod(2:N₁) do i
        return space(N, i)
    end
    PNp = prod(2:N₁) do i
        return flip(space(N, i))
    end
    return isomorphism(storagetype(N), PNp, PN)
end
@non_differentiable physical_flipper(args...)

@generated function apply_physical_unitary(
    N::MPSKit.GenericMPSTensor{S,N₁}, U::SquareTensorMap{S,N₂}
) where {S,N₁,N₂}
    N_out_e = tensorexpr(:N_out, -(1:N₁), -(N₁ + 1))
    N_e = tensorexpr(:N, (-1, (1:N₂)...), -(N₁ + 1))
    U_e = tensorexpr(:U, -(2:N₁), (1:N₂))
    return macroexpand(@__MODULE__, :(return @tensor $N_out_e := $N_e * $U_e))
end

function physical_flip(A::MPSKit.GenericMPSTensor{S,N₁}) where {S,N₁}
    # return apply_physical_unitary(A, physical_flipper(A))
    return flip(A, 2:N₁)
end

# short-circuit this
function ChainRulesCore.rrule(::typeof(physical_flip), A::MPSKit.GenericMPSTensor)
    Ap = physical_flip(A)

    function physical_flip_pullback(ΔAp)
        ΔAp = unthunk(ΔAp)
        return NoTangent(), physical_flip(ΔAp)
    end
    return Ap, physical_flip_pullback
end

function LinearAlgebra.schur(t::TensorMap; kwargs...)
    return LinearAlgebra.schur!(copy(t); kwargs...)
end
function LinearAlgebra.schur!(t::TensorMap; kwargs...)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`schur!` requires domain and codomain to be the same"))

    I = sectortype(t)
    S = spacetype(t)
    dims = SectorDict{I,Int}(c => size(b, 1) for (c, b) in blocks(t))
    W = S(dims)

    T = similar(t, W ← W)
    Z = similar(t, domain(t) ← W)
    values = SectorDict{I,Vector{scalartype(t)}}()
    for (c, b) in blocks(t)
        Tb, Zb, valb = LinearAlgebra.schur!(collect(b); kwargs...)
        copy!(block(T, c), Tb)
        copy!(block(Z, c), Zb)
        values[c] = valb
    end
    return T, Z, values
end

function _conj(A::MPSKit.GenericMPSTensor{S,N₁}) where {S,N₁}
    return permute(A', ((1, (3:(N₁ + 1))...), (2,)))
end

# trying to optimize the unitary flipper

# skipping this altogether for now, but keeping it around for later

function get_unitary_costfun(Q, N, W)
    function f(U)
        E, gs = withgradient(U) do Up
            LHS = transfer_left(Q, N, W, Up)
            λ = tr(LHS * Q')
            return -log(abs(λ))
        end
        Δ = TKM.Unitary.project!(only(gs), U)
        return E, Δ
    end
    return f
end

function optimize_physical_unitary(
    N,
    W;
    Q0=MPSKit.randomize!(similar(N, space(W, 1) ← space(N, 1))),
    Up0::SquareTensorMap=physical_isometry(env.N, env.W),
    tol_conv=Defaults.ctmrg_tol,
    tol_eigs=MPSKit.Defaults.tol,
    maxiter=1000,
    verbosity=1,
)
    λ, Q = MPSKit.fixedpoint(gen_transfer_left(N, W, Up0), Q0, :LM; tol=tol_eigs)
    u, _, v = tsvd(Q)
    Q = u * v

    # if the leading eigenvalue is not a phase, iteratively update the physical map until it is
    iter = 0
    Up = Up0

    while !isapprox(abs(λ), 1.0; atol=tol_conv) && iter < maxiter
        iter += 1
        verbosity > 1 && @info "Symmetrization at iter=$iter: abs(λ)=$(abs(λ))"

        # proper optimization update, 'works' but don't know if the result is any good...
        Up, = optimize(
            get_unitary_costfun(Q, N, W),
            Up,
            LBFGS(; maxiter=5, gradtol=tol_eigs, verbosity=verbosity - 1);
            inner=TKM.Unitary.inner,
            retract=TKM.Unitary.retract,
            (transport!)=(TKM.Unitary.transport!),
        )

        λ, Q = MPSKit.fixedpoint(gen_transfer_left(N, W, Up), Q, :LM; tol=tol_eigs)
    end
    verbosity > 0 && @info "Symmetrization terminated at iter=$iter with abs(λ)=$(abs(λ))"

    u, _, v = tsvd(Q)
    Q = u * v

    return Up, Q
end

#
# Symmetric contraction environment
#

"""
    struct SymmetricEnv{C,T}
"""
struct SymmetricEnv{C<:MPSKit.MPSBondTensor,T<:MPSKit.GenericMPSTensor}
    X::C
    A::T
end

function symmetric_environment(::Val{:center}, args...; kwargs...)
    return center_gauge_environment(args...; kwargs...)
end

function symmetric_environment(::Val{:left}, args...; kwargs...)
    return left_gauge_environment(args...; kwargs...)
end

function diagonalize_corner(
    X::MPSKit.MPSBondTensor{S},
    N::MPSKit.GenericMPSTensor{S,N₁},
    W::MPSKit.GenericMPSTensor{S,N₁},
) where {S,N₁}
    u, X, v = tsvd(X, (1,), (2,))
    N = absorb_bond_unitary(N, v)
    W = absorb_bond_unitary(W, u')
    return X, N, W
end

function match_edges(
    X::MPSKit.MPSBondTensor{S},
    N::MPSKit.GenericMPSTensor{S,N₁},
    W::MPSKit.GenericMPSTensor{S,N₁},
    Up0::SquareTensorMap=physical_isometry(N, W); # TODO: update to use externally supplied physical unitary
    tol_conv=Defaults.ctmrg_tol,
    tol_eigs=MPSKit.Defaults.tol,
    maxiter=1000,
    verbosity=1,
) where {S,N₁}
    # find the left fixed point of the generalized left transfer matrix
    Q0 = MPSKit.randomize!(similar(N, space(W, 1) ← space(N, 1)))
    λ, Q = MPSKit.fixedpoint(gen_transfer_left(N, W, Up0), Q0, :LM; tol=tol_eigs)
    u, _, v = tsvd(Q)
    Q = u * v

    isapprox(abs(λ), 1.0; atol=tol_conv) ||
        @warn "Requiring physical unitary other than the spaceflip, probably something went wrong"

    # Up, Q = optimize_physical_unitary(N, W; Q0, Up0, tol_conv, tol_eigs, maxiter, verbosity)
    Up = Up0 # skip physical unitary optimization for now

    # absorb the unitary into the corner tensor and update the north and west edges
    Xm, V, = schur(Q' * X)
    Nm = absorb_bond_unitary(N, V')
    Wm = absorb_bond_unitary(W, (Q * V)')

    # one more time, just to be sure...
    λ, = MPSKit.fixedpoint(gen_transfer_left(Nm, Wm, Up), Q, :LM; tol=tol_eigs)

    return Xm, Nm, Wm, Up, λ
end

function normalize_corner(X::MPSKit.MPSBondTensor{S}; tol=1e-14) where {S}
    # impose normalization on X
    X /= (tr(X^4)^(1 / 4))

    # get rid of spurious phases caused by root
    fisrt_element = first(first(diag(X))[2])
    phase = round(Int, angle(fisrt_element) * 2 / pi) % 4
    if phase != 0
        X = exp(im * phase * pi / 2) * X
    end

    # make corner tensor to be diagonal
    Xd = DiagonalTensorMap(X)
    norm(X - Xd) < tol ||
        @warn "Corner tensor is not diagonal enough: norm(X - diag(X))=$(norm(X - Xd))"
    X = Xd

    # make corner tensor real
    norm(imag(X)) < tol ||
        @warn "Corner tensor is not real enough: norm(imag(X))=$(norm(imag(X)))"
    X = real(X)

    return X
end

function center_gauge_environment(
    env::PullingThroughEnv;
    tol_conv=Defaults.ctmrg_tol,
    tol_eigs=MPSKit.Defaults.tol,
    maxiter=1000,
    verbosity=1,
)
    # check convergence and unpack
    norm(env.LN - env.RW) < tol_conv ||
        @warn "Pulling through contraction is not converged!"
    N = env.N
    W = env.W
    X = env.LN

    # diagonalize corner tensor
    X, N, W = diagonalize_corner(X, N, W)

    # match north and west edges
    Xm, Nm, Wm, Up, λ = match_edges(X, N, W; tol_conv, tol_eigs, maxiter, verbosity)
    # check if the edges are now actually the same
    ovlp = tr(Nm' * PEPSKit.apply_physical_unitary(Wm, Up')) / (norm(Nm) * norm(Wm))
    isapprox(abs(ovlp), 1.0; atol=tol_conv) ||
        @warn "Edges are not the same after symmetrization: abs(ovlp)=$(abs(ovlp))"

    # impose normalization on X
    Xm = normalize_corner(Xm; tol=tol_conv)

    # check hermiticity of Nm
    N̄m = physical_flip(_conj(Nm)) # TODO: update to use externally supplied physical unitary
    if norm(Nm - N̄m) > tol_conv
        phase = angle(dot(Nm, N̄m))
        scale!(Nm, exp(im * phase / 2))
        N̄m = physical_flip(_conj(Nm))
        norm(Nm - N̄m) < tol_conv ||
            @warn "North edge is not hermitian enough: norm(Nm - N̄m)=$(norm(Nm - N̄m))"
    end
    Nm = normalize_mps(Nm) # just to be absolutely safe

    return SymmetricEnv(Xm, Nm), Up, Wm, λ
end

function left_gauge_environment(
    env::PullingThroughEnv;
    tol_conv=Defaults.ctmrg_tol,
    tol_eigs=MPSKit.Defaults.tol,
    maxiter=1000,
    verbosity=1,
)
    # check convergence and unpack
    norm(env.LN - env.RW) < tol_conv ||
        @warn "Pulling through contraction is not converged!"
    N = env.NL
    W = _conj(env.WR)
    # @show norm(W' * W - id(domain(W))) # this is actually a left isometry, which is good
    X = env.LN

    # diagonalize corner tensor
    X, N, W = diagonalize_corner(X, N, W)

    # match north and west edges
    Xm, Nm, Wm, Up, λ = match_edges(X, N, W; tol_conv, tol_eigs, maxiter, verbosity)
    # check if the edges are now actually the same
    ovlp = tr(Nm' * PEPSKit.apply_physical_unitary(Wm, Up')) / (norm(Nm) * norm(Wm))
    isapprox(abs(ovlp), 1.0; atol=tol_conv) ||
        @warn "Edges are not the same after symmetrization: abs(ovlp)=$(abs(ovlp))"

    # impose normalization on X
    Xm = normalize_corner(Xm; tol=tol_conv)

    # check hermiticity of Nm
    N̄m = physical_flip(_conj(Nm)) # TODO: update to use externally supplied physical unitary
    C2N̄m = absorb_left_bond_matrix(N̄m, Xm^2)
    NmC2 = absorb_right_bond_matrix(Nm, Xm^2)
    if norm(C2N̄m - NmC2) > tol_conv
        phase = angle(dot(Nm, N̄m))
        scale!(Nm, exp(im * phase / 2))
        C2N̄m = absorb_left_bond_matrix(physical_flip(_conj(Nm)), Xm^2)
        NmC2 = absorb_right_bond_matrix(Nm, Xm^2)
        norm(NmC2 - C2N̄m) < tol_conv ||
            @warn "North edge is not hermitian enough: norm(Nm - N̄m)=$(norm(NmC2 - C2N̄m))"
    end
    Nm = normalize_mps(Nm) # just to be absolutely safe

    return SymmetricEnv(Xm, Nm), Up, Wm, λ
end

function Base.complex(env::SymmetricEnv)
    return SymmetricEnv(complex(env.X), complex(env.A))
end

# to be able to start iterating again with a symmetrized initial guess
function PullingThroughEnv(env::SymmetricEnv)
    # flip west, seed gauging matrices with corner tensor
    return PullingThroughEnv(env.A, physical_flip(env.A); LN=env.X, RW=env.X)
end

# In-place update of environment
function update!(env::SymmetricEnv{C,T}, env´::SymmetricEnv{C,T}) where {C,T}
    copy!(env.A, env´.A)
    copy!(env.X, env´.X)
    return env
end

function VI.scalartype(::Type{SymmetricEnv{C,T}}) where {C,T}
    S₁ = scalartype(C)
    S₂ = scalartype(T)
    return promote_type(S₁, S₂)
end
