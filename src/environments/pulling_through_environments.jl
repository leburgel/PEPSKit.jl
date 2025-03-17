
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

# symmetric pulling through, only use a single virtual space

function PullingThroughEnv(st::InfiniteSquareNetwork, chi::ElementarySpaceLike)
    return PullingThroughEnv(randn, ComplexF64, st, chi)
end
function PullingThroughEnv(
    f, ::Type{T}, peps::InfinitePEPS, chi::ElementarySpaceLike
) where {T}
    chi = _to_space(chi)
    P = only(peps.A)
    D_N_above = adjoint(space(P, 2))
    D_N_below = adjoint(D_N_above)
    D_W_above = adjoint(space(P, 5))
    D_W_below = adjoint(D_W_above)

    N = TensorMap(f, T, chi ⊗ D_N_above ⊗ D_N_below ← chi)
    W = TensorMap(f, T, chi ⊗ D_W_above ⊗ D_W_below ← chi)

    return PullingThroughEnv(N, W)
end
function PullingThroughEnv(
    f, ::Type{T}, partfunc::InfinitePartitionFunction, chi::ElementarySpaceLike
) where {T}
    chi = _to_space(chi)
    Z = only(partfunc.A)
    D_N = adjoint(space(Z, 3))
    D_W = adjoint(space(Z, 1))

    N = TensorMap(f, T, chi ⊗ D_N ← chi)
    W = TensorMap(f, T, chi ⊗ D_W ← chi)

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
        alg_orth=alg_gauge.alg_orth,
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
    absorb_bond_unitary(A, U)

Absorb a bond unitary into an MPS tensor.

```
 ←A←  <--  ←U←A←U'←
  ↓           ↓
```
"""
@generated function absorb_bond_unitary(
    A::MPSKit.GenericMPSTensor{S,N₁}, U::MPSKit.MPSBondTensor{S}
) where {S,N₁}
    A_out_e = tensorexpr(:A_out, -(1:N₁), -(N₁ + 1))
    A_e = tensorexpr(:A, (1, (-(2:N₁))...), 2)
    Ū_e = tensorexpr(:U, -(N₁ + 1), 2)
    U_e = tensorexpr(:U, -1, 1)
    return macroexpand(@__MODULE__, :(return @tensor $A_out_e := $U_e * $A_e * conj($Ū_e)))
end

"""
    absorb_bond_matrices(A, X1, X2)

Absorb left and right bond matrices into an MPS tensor.

```
 ←A←  <--  ←X1←A←X2←
  ↓            ↓
```
"""
@generated function absorb_bond_matrices(
    A::MPSKit.GenericMPSTensor{S,N₁},
    X1::MPSKit.MPSBondTensor{S},
    X2::MPSKit.MPSBondTensor{S},
) where {S,N₁}
    A_out_e = tensorexpr(:A_out, -(1:N₁), -(N₁ + 1))
    XL_e = tensorexpr(:X1, -1, 1)
    A_e = tensorexpr(:A, (1, (-(2:N₁))...), 2)
    XR_e = tensorexpr(:X2, 2, -(N₁ + 1))
    return macroexpand(@__MODULE__, :(return @tensor $A_out_e := $XL_e * $A_e * $XR_e))
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
    return apply_physical_unitary(A, physical_flipper(A))
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

function ChainRulesCore.rrule(
    ::typeof(physical_flipper), N::MPSKit.GenericMPSTensor{S,N₁}
) where {S,N₁}
    return N, ΔN -> (ΔN,)
end

# trying to optimize the unitary flipper, and failing
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

# all the tricks to symmetrize the environment
function symmetric_environment(
    env::PullingThroughEnv,
    Up0::SquareTensorMap=physical_isometry(env.N, env.W);
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
    u, X, v = tsvd(X, (1,), (2,))
    N = absorb_bond_unitary(N, v)
    W = absorb_bond_unitary(W, u')

    # match north and west edges:
    # find the left fixed point of the generalized left transfer matrix
    Q0 = MPSKit.randomize!(similar(N, space(W, 1) ← space(N, 1)))
    λ, Q = MPSKit.fixedpoint(gen_transfer_left(N, W, Up0), Q0, :LM; tol=tol_eigs)
    u, _, v = tsvd(Q)
    Q = u * v

    # if the leading eigenvalue is not a phase, iteratively update the physical map until it is
    iter = 0
    Up = Up0

    isapprox(abs(λ), 1.0; atol=tol_conv) ||
        @warn "Requiring physical unitary other than the spaceflip, probably something went wrong"

    # TODO: get rid of this whole thing, and just add virtual flippers everywhere...
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

    # absorb the unitary into the corner tensor and update the north and west edges
    Xm, V, = schur(Q' * X) # TODO: figure out purely imaginary Xm?
    Nm = absorb_bond_unitary(N, V')
    Wm = absorb_bond_unitary(W, (Q * V)')

    # one more time, just to be sure...
    λ, = MPSKit.fixedpoint(gen_transfer_left(Nm, Wm, Up), Q, :LM; tol=tol_eigs)

    # impose normalization on X
    Xm = Xm / (tr(Xm^4)^(1 / 4)) # TODO: daggers or not?

    # impose hermiticity on Nm
    Nm_dag = _conj(Nm)
    N̄m = physical_flip(Nm_dag) # TODO: we probably want to apply the actual physical unitary if there is one...
    Nm = (Nm + N̄m) / 2
    Nm = normalize_mps(Nm)

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

# Custom adjoint for SymmetricEnv constructor, needed for fixed-point differentiation
function ChainRulesCore.rrule(::Type{SymmetricEnv}, X, A)
    env = SymmetricEnv(X, A)
    function symmetricenv_pullback(Δenv)
        Δenv = unthunk(Δenv)
        return NoTangent(), Δenv.X, Δenv.A
    end
    return env, symmetricenv_pullback
end

# Custom adjoint for SymmetricEnv getproperty, to avoid creating named tuples in backward pass
function ChainRulesCore.rrule(::typeof(getproperty), e::SymmetricEnv, name::Symbol)
    result = getproperty(e, name)
    if name === :X
        function corner_pullback(ΔX)
            ΔX = unthunk(ΔX)
            return NoTangent(), SymmetricEnv(ΔX, zerovector(e.A)), NoTangent()
        end
        return result, corner_pullback
    elseif name === :A
        function edge_pullback(ΔA)
            ΔA = unthunk(ΔA)
            return NoTangent(), SymmetricEnv(zerovector(e.X), ΔA), NoTangent()
        end
        return result, edge_pullback
    else
        # this should never happen because already errored in forwards pass
        throw(ArgumentError("No rrule for getproperty of $name"))
    end
end

# Functions used for FP differentiation and by KrylovKit.linsolve
function Base.:+(e₁::SymmetricEnv, e₂::SymmetricEnv)
    return SymmetricEnv(e₁.X + e₂.X, e₁.A + e₂.A)
end
function Base.:-(e₁::SymmetricEnv, e₂::SymmetricEnv)
    return SymmetricEnv(e₁.X - e₂.X, e₁.A - e₂.A)
end
Base.:*(α::Number, e::SymmetricEnv) = SymmetricEnv(α * e.X, α * e.A)
Base.:*(e::SymmetricEnv, α::Number) = α * e
Base.similar(e::SymmetricEnv) = SymmetricEnv(similar(e.X), similar(e.A))

function LinearAlgebra.mul!(edst::SymmetricEnv, esrc::SymmetricEnv, α::Number)
    mul!(edst.X, esrc.X, α)
    mul!(edst.A, esrc.A, α)
    return edst
end

function LinearAlgebra.rmul!(e::SymmetricEnv, α::Number)
    rmul!(e.X, α)
    rmul!(e.A, α)
    return e
end

function LinearAlgebra.axpy!(α::Number, e₁::SymmetricEnv, e₂::SymmetricEnv)
    axpy!(α, e₁.X, e₂.X)
    axpy!(α, e₁.A, e₂.A)
    return e₂
end

function LinearAlgebra.axpby!(α::Number, e₁::SymmetricEnv, β::Number, e₂::SymmetricEnv)
    axpby!(α, e₁.X, β, e₂.X)
    axpby!(α, e₁.A, β, e₂.A)
    return e₂
end

function LinearAlgebra.dot(e₁::SymmetricEnv, e₂::SymmetricEnv)
    return dot(e₁.X, e₂.X) + dot(e₁.A, e₂.A)
end

# VectorInterface
# ---------------

# Note: the following methods consider the environment tensors as separate components of one
# big vector. In other words, the associated vector space is not the natural one associated
# to the original (physical) system, and addition, scaling, etc. are performed element-wise.

import VectorInterface as VI

function VI.scalartype(::Type{SymmetricEnv{C,T}}) where {C,T}
    S₁ = scalartype(C)
    S₂ = scalartype(T)
    return promote_type(S₁, S₂)
end

function VI.zerovector(env::SymmetricEnv, ::Type{S}) where {S<:Number}
    _zerovector = Base.Fix2(zerovector, S)
    return SymmetricEnv(_zerovector(env.X), _zerovector(env.A))
end
function VI.zerovector!(env::SymmetricEnv)
    zerovector!(env.X)
    zerovector!(env.A)
    return env
end
VI.zerovector!!(env::SymmetricEnv) = zerovector!(env)

function VI.scale(env::SymmetricEnv, α::Number)
    _scale = Base.Fix2(scale, α)
    return SymmetricEnv(_scale(env.X), _scale(env.A))
end
function VI.scale!(env::SymmetricEnv, α::Number)
    _scale! = Base.Fix2(scale!, α)
    _scale!(env.X)
    _scale!(env.A)
    return env
end
function VI.scale!(env₁::SymmetricEnv, env₂::SymmetricEnv, α::Number)
    _scale!(x, y) = scale!(x, y, α)
    _scale!(env₁.X, env₂.X)
    _scale!(env₁.A, env₂.A)
    return env₁
end
VI.scale!!(env::SymmetricEnv, α::Number) = scale!(env, α)
VI.scale!!(env₁::SymmetricEnv, env₂::SymmetricEnv, α::Number) = scale!(env₁, env₂, α)

function VI.add(env₁::SymmetricEnv, env₂::SymmetricEnv, α::Number, β::Number)
    _add(x, y) = add(x, y, α, β)
    return SymmetricEnv(_add(env₁.X, env₂.X), _add(env₁.A, env₂.A))
end
function VI.add!(env₁::SymmetricEnv, env₂::SymmetricEnv, α::Number, β::Number)
    _add!(x, y) = add!(x, y, α, β)
    _add!(env₁.X, env₂.X)
    _add!(env₁.A, env₂.A)
    return env₁
end
function VI.add!!(env₁::SymmetricEnv, env₂::SymmetricEnv, α::Number, β::Number)
    return add!(env₁, env₂, α, β)
end

# Exploiting the fact that VectorInterface works for tuples:
function VI.inner(env₁::SymmetricEnv, env₂::SymmetricEnv)
    return inner((env₁.X, env₁.A), (env₂.X, env₂.A))
end
VI.norm(env::SymmetricEnv) = norm((env.X, env.A))
