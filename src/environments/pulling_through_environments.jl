
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
        tol=alg_gauge.tol, maxiter=alg_gauge.maxiter, verbosity=alg_gauge.verbosity
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
        tol=alg_gauge.tol, maxiter=alg_gauge.maxiter, verbosity=alg_gauge.verbosity
    )
    AR = MPSKit.PeriodicArray([copy(W)])
    C = MPSKit.PeriodicArray([copy(RW0)])

    MPSKit.uniform_rightorth!((AR, C), [W], RW0, rightorth_alg)

    @reset env.WR = only(AR)
    @reset env.RW = only(C)

    return env
end
