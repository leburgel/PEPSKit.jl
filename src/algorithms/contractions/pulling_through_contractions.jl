#
# Auxiliary contractions
#

"""
    absorb_bond_unitary(A, U)

Absorb a bond unitary into an MPS tensor.

```
 ←A←  <--  ←U←A←U'←
  ↓           ↓
```
"""
function absorb_bond_unitary(
    A::MPSKit.GenericMPSTensor{S,N₁}, U::MPSKit.MPSBondTensor{S}
) where {S,N₁}
    return absorb_bond_matrices(A, U, U')
end

"""
    absorb_left_bond_matrix(A, X)

Absorb left bond matrix into an MPS tensor.

```
 ←A←  <--  ←X←A←
  ↓           ↓
```
"""
function absorb_left_bond_matrix(
    A::MPSKit.GenericMPSTensor{S}, X::MPSKit.MPSBondTensor{S}
) where {S}
    pX = (codomainind(X), domainind(X))
    pA = ((codomainind(A)[1],), (codomainind(A)[2:end]..., domainind(A)...))
    pXA = (codomainind(A), domainind(A))
    return tensorcontract(X, pX, false, A, pA, false, pXA)
end

"""
    absorb_right_bond_matrix(A, X)

Absorb right bond matrix into an MPS tensor.

```
 ←A←  <--  ←A←X←
  ↓         ↓
```
"""
function absorb_right_bond_matrix(
    A::MPSKit.GenericMPSTensor{S}, X::MPSKit.MPSBondTensor{S}
) where {S}
    return A * X
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

#
# Pulling through iteration contractions
#

# PEPS contractions

function transfer_north(
    N::MPSKit.GenericMPSTensor{S,3}, WR::MPSKit.GenericMPSTensor{S,3}, O::PEPSSandwich
) where {S}
    return @autoopt @tensor N´[χ_SW D_S_above D_S_below; χ_SE] :=
        WR[χ_SW D_W_above D_W_below; χ_NW] *
        N[χ_NW D_N_above D_N_below; χ_NE] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below]) *
        conj(WR[χ_SE, D_E_above, D_E_below; χ_NE])
end

function transfer_west(
    W::MPSKit.GenericMPSTensor{S,3}, NL::MPSKit.GenericMPSTensor{S,3}, O::PEPSSandwich
) where {S}
    return @autoopt @tensor W´[χ_SE D_E_above D_E_below; χ_NE] :=
        NL[χ_NW D_N_above D_N_below; χ_NE] *
        W[χ_SW D_W_above D_W_below; χ_NW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below]) *
        conj(NL[χ_SW, D_S_above, D_S_below; χ_SE])
end

# partition function contractions

function transfer_north(
    N::MPSKit.GenericMPSTensor{S,2},
    WR::MPSKit.GenericMPSTensor{S,2},
    partfunc::PartitionFunctionTensor,
) where {S}
    return @autoopt @tensor N´[χ_SW D_S; χ_SE] :=
        WR[χ_SW D_W; χ_NW] *
        N[χ_NW D_N; χ_NE] *
        partfunc[D_W D_S; D_N D_E] *
        conj(WR[χ_SE, D_E; χ_NE])
end

function transfer_west(
    W::MPSKit.GenericMPSTensor{S,2},
    NL::MPSKit.GenericMPSTensor{S,2},
    partfunc::PartitionFunctionTensor,
) where {S}
    return @autoopt @tensor W´[χ_SE D_E; χ_NE] :=
        NL[χ_NW D_N; χ_NE] *
        W[χ_SW D_W; χ_NW] *
        partfunc[D_W D_S; D_N D_E] *
        conj(NL[χ_SW, D_S; χ_SE])
end

# PEPO contractions: TODO

#
# Pulling through fixed-point contractions
#

## FP1: hermiticity

# duplicate arguments for use in forward computation
function fixed_point_1_id(
    ::Val{Gauge}, ::Val{Style}, A::MPSKit.GenericMPSTensor{S}, X::MPSKit.MPSBondTensor{S}
) where {Gauge,Style,S}
    return fixed_point_1_id(Val(Gauge), Val(Style), A, X, X)
end

# expanded calls for use in derivatives
function fixed_point_1_id(
    ::Val{:center},
    ::Val{:naive},
    A::MPSKit.GenericMPSTensor{S},
    X1::MPSKit.MPSBondTensor{S},
    X2::MPSKit.MPSBondTensor{S},
) where {S}
    return A
end
function fixed_point_1_id(
    ::Val{:center},
    ::Val{:regularized},
    A::MPSKit.GenericMPSTensor{S},
    X1::MPSKit.MPSBondTensor{S},
    X2::MPSKit.MPSBondTensor{S},
) where {S}
    return absorb_bond_matrices(A, X1, X2)
end
function fixed_point_1_id(
    ::Val{:left},
    ::Val{Style},
    A::MPSKit.GenericMPSTensor{S},
    X1::MPSKit.MPSBondTensor{S},
    X2::MPSKit.MPSBondTensor{S},
) where {Style,S}
    return absorb_right_bond_matrix(A, X1 * X2)
end

# duplicate arguments for use in forward computation
function fixed_point_1_conj(
    ::Val{Gauge}, ::Val{Style}, A::MPSKit.GenericMPSTensor{S}, X::MPSKit.MPSBondTensor{S}
) where {Gauge,Style,S}
    return fixed_point_1_conj(Val(Gauge), Val(Style), A, X, X)
end

# expanded calls for use in derivatives
function fixed_point_1_conj(
    ::Val{:center},
    ::Val{:naive},
    A::MPSKit.GenericMPSTensor{S},
    X1::MPSKit.MPSBondTensor{S},
    X2::MPSKit.MPSBondTensor{S},
) where {S}
    return physical_flip(_conj(A))
end
function fixed_point_1_conj(
    ::Val{:center},
    ::Val{:regularized},
    A::MPSKit.GenericMPSTensor{S},
    X1::MPSKit.MPSBondTensor{S},
    X2::MPSKit.MPSBondTensor{S},
) where {S}
    return absorb_bond_matrices(physical_flip(_conj(A)), X1, X2)
end
function fixed_point_1_conj(
    ::Val{:left},
    ::Val{Style},
    A::MPSKit.GenericMPSTensor{S},
    X1::MPSKit.MPSBondTensor{S},
    X2::MPSKit.MPSBondTensor{S},
) where {Style,S}
    return absorb_left_bond_matrix(physical_flip(_conj(A)), X1 * X2)
end

function fixed_point_1(
    ::Val{Style}, ::Val{Gauge}, A::MPSKit.GenericMPSTensor{S}, X::MPSKit.MPSBondTensor{S}
) where {Style,Gauge,S}
    FP1 =
        fixed_point_1_id(Val(Style), Val(Gauge), A, X) -
        fixed_point_1_conj(Val(Style), Val(Gauge), A, X)
    return FP1
end

## FP2: eigenvalue equation

# start by defining fixed-point transfer functions

# these flip the physical space of the west and south edges in the input, consistent with
# our implicit assumption that the PEPS flip bond tensors have been absorbed into the west
# and south virtual spaces of the PEPS tensors.

# duplicate arguments for use in forward computation
function fixed_point_2_transfer(
    ::Val{Gauge}, ::Val{Style}, A::MPSKit.GenericMPSTensor{S}, X::MPSKit.MPSBondTensor{S}, O
) where {Gauge,Style,S}
    return fixed_point_2_transfer(Val(Gauge), Val(Style), A, A, A, X, X, O)
end

# expanded call for use in backward pass
function fixed_point_2_transfer(
    ::Val{Gauge},
    ::Val{Style},
    AN::MPSKit.GenericMPSTensor{S,3},
    AW::MPSKit.GenericMPSTensor{S,3},
    AS::MPSKit.GenericMPSTensor{S,3},
    XNW::MPSKit.MPSBondTensor{S},
    XSW::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
) where {Gauge,Style,S}
    AWU = physical_flip(AW) # TODO: get rid of this
    ASU = physical_flip(AS) # TODO: get rid of this
    @autoopt @tensor AW´[χ_SE D_E_above D_E_below; χ_NE] :=
        AN[χ_NNW D_N_above D_N_below; χ_NE] *
        AWU[χ_WSW D_W_above D_W_below; χ_WNW] *
        ASU[χ_SE D_S_above D_S_below; χ_SSW] *
        XNW[χ_WNW; χ_NNW] *
        XSW[χ_SSW; χ_WSW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
    return physical_flip(AW´) # restore original space...
end
function fixed_point_2_transfer(
    ::Val{Gauge},
    ::Val{Style},
    AN::MPSKit.GenericMPSTensor{S,2},
    AW::MPSKit.GenericMPSTensor{S,2},
    AS::MPSKit.GenericMPSTensor{S,2},
    XNW::MPSKit.MPSBondTensor{S},
    XSW::MPSKit.MPSBondTensor{S},
    O::PFTensor,
) where {Gauge,Style,S}
    AWU = physical_flip(AW) # TODO: get rid of this
    ASU = physical_flip(AS) # TODO: get rid of this
    @autoopt @tensor AW´[χ_SE D_E; χ_NE] :=
        AN[χ_NNW D_N; χ_NE] *
        AWU[χ_WSW D_W; χ_WNW] *
        ASU[χ_SE D_S; χ_SSW] *
        XNW[χ_WNW; χ_NNW] *
        XSW[χ_SSW; χ_WSW] *
        O[D_W D_S; D_N D_E]
    return physical_flip(AW´) # restore original space...
end

# duplicate arguments for use in forward computation
function fixed_point_2_vector(
    ::Val{Gauge},
    ::Val{Style},
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    N::Number,
) where {Gauge,Style,S}
    return fixed_point_2_vector(Val(Gauge), Val(Style), A, X, X, N)
end

# expanded call for use in backward pass
function fixed_point_2_vector(
    ::Val{Gauge},
    ::Val{Style},
    AW::MPSKit.GenericMPSTensor{S},
    XNW::MPSKit.MPSBondTensor{S},
    XSW::MPSKit.MPSBondTensor{S},
    N::Number,
) where {Gauge,Style,S}
    return N * absorb_bond_matrices(AW, XSW, XNW)
end

function fixed_point_2(
    ::Val{Gauge},
    ::Val{Style},
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    N::Number,
    O,
) where {Gauge,Style,S}
    FP2 =
        fixed_point_2_transfer(Val(Gauge), Val(Style), A, X, O) -
        fixed_point_2_vector(Val(Gauge), Val(Style), A, X, N)
    return FP2
end

## FP3: boundary normalization (isometry) condition

# duplicate arguments for use in forward computation
function fixed_point_3_transfer(
    ::Val{Gauge}, ::Val{Style}, A::MPSKit.GenericMPSTensor{S}, X::MPSKit.MPSBondTensor{S}
) where {Gauge,Style,S}
    return fixed_point_3_transfer(Val(Gauge), Val(Style), A, A, X, X)
end

# expanded calls for use in derivatives
function fixed_point_3_transfer(
    ::Val{:center},
    ::Val{Style},
    AN::MPSKit.GenericMPSTensor{S},
    AS::MPSKit.GenericMPSTensor{S},
    XNW::MPSKit.MPSBondTensor{S},
    XSW::MPSKit.MPSBondTensor{S},
) where {Style,S}
    FP3 = MPSKit.transfer_left(XSW' * XNW, AN, AS)
    return FP3
end
function fixed_point_3_transfer(
    ::Val{:left},
    ::Val{:naive},
    AN::MPSKit.GenericMPSTensor{S},
    AS::MPSKit.GenericMPSTensor{S},
    ::MPSKit.MPSBondTensor{S},
    ::MPSKit.MPSBondTensor{S},
) where {S}
    FP3 = AS' * AN
    return FP3
end
function fixed_point_3_transfer(
    ::Val{:left},
    ::Val{:regularized},
    AN::MPSKit.GenericMPSTensor{S},
    AS::MPSKit.GenericMPSTensor{S},
    XNE::MPSKit.MPSBondTensor{S},
    XSE::MPSKit.MPSBondTensor{S},
) where {S}
    FP3 = XSE' * (AS' * AN) * XNE
    return FP3
end

function fixed_point_3_vector(
    ::Val{Gauge}, ::Val{Style}, X::MPSKit.MPSBondTensor{S}
) where {Gauge,Style,S}
    return fixed_point_3_vector(Val(Gauge), Val(Style), X, X)
end
function fixed_point_3_vector(
    ::Val{:center}, ::Val{Style}, XNE::MPSKit.MPSBondTensor{S}, XSE::MPSKit.MPSBondTensor{S}
) where {Style,S}
    return XSE' * XNE
end
function fixed_point_3_vector(
    ::Val{:left}, ::Val{:naive}, XNE::MPSKit.MPSBondTensor{S}, XSE::MPSKit.MPSBondTensor{S}
) where {S}
    return id(codomain(XNE))
end
function fixed_point_3_vector(
    ::Val{:left},
    ::Val{:regularized},
    XNE::MPSKit.MPSBondTensor{S},
    XSE::MPSKit.MPSBondTensor{S},
) where {S}
    return XSE' * XNE
end

function fixed_point_3(
    ::Val{Gauge}, ::Val{Style}, A::MPSKit.GenericMPSTensor{S}, X::MPSKit.MPSBondTensor{S}
) where {Gauge,Style,S}
    FP3 =
        fixed_point_3_transfer(Val(Gauge), Val(Style), A, X) -
        fixed_point_3_vector(Val(Gauge), Val(Style), X)
    return FP3
end

## FP4: corner normalization normalization condition

function fixed_point_4(
    ::Val{Gauge}, ::Val{Style}, X::MPSKit.MPSBondTensor{S}
) where {Gauge,Style,S}
    return tr(X^4) - one(scalartype(X))
end

#
# Pulling through pushforward contractions
#

## FP1: hermiticity

function fixed_point_1_pushforward(
    ::Val{:center},
    ::Val{:naive},
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    ∂A::MPSKit.GenericMPSTensor{S},
    ∂X::MPSKit.MPSBondTensor{S},
) where {S}
    ∂FP1 =
        fixed_point_1_id(Val(:center), Val(:naive), ∂A, X, X) -
        fixed_point_1_conj(Val(:center), Val(:naive), ∂A, X, X)
    return ∂FP1
end
function fixed_point_1_pushforward(
    ::Val{:center},
    ::Val{:regularized},
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    ∂A::MPSKit.GenericMPSTensor{S},
    ∂X::MPSKit.MPSBondTensor{S},
) where {S}
    ∂FP1 =
        fixed_point_1_id(Val(:center), Val(:regularized), ∂A, X, X) +
        fixed_point_1_id(Val(:center), Val(:regularized), A, ∂X, X) +
        fixed_point_1_id(Val(:center), Val(:regularized), A, X, ∂X) -
        fixed_point_1_conj(Val(:center), Val(:regularized), ∂A, X, X) -
        fixed_point_1_conj(Val(:center), Val(:regularized), ∂A, X, X) -
        fixed_point_1_conj(Val(:center), Val(:regularized), ∂A, X, X)
    return ∂FP1
end
function fixed_point_1_pushforward(
    ::Val{:left},
    ::Val{Style},
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    ∂A::MPSKit.GenericMPSTensor{S},
    ∂X::MPSKit.MPSBondTensor{S},
) where {Style,S}
    ∂FP1 =
        fixed_point_1_id(Val(:left), Val(Style), ∂A, X, X) +
        fixed_point_1_id(Val(:left), Val(Style), A, ∂X, X) +
        fixed_point_1_id(Val(:left), Val(Style), A, X, ∂X) -
        fixed_point_1_conj(Val(:left), Val(Style), ∂A, X, X) -
        fixed_point_1_conj(Val(:left), Val(Style), ∂A, X, X) -
        fixed_point_1_conj(Val(:left), Val(Style), ∂A, X, X)
    return ∂FP1
end

## FP2: eigenvalue equation

function fixed_point_2_pushforward(
    ::Val{Gauge},
    ::Val{Style},
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    N::Number,
    O::PEPSSandwich,
    ∂A::MPSKit.GenericMPSTensor{S},
    ∂X::MPSKit.MPSBondTensor{S},
    ∂N::Number,
) where {Gauge,Style,S}
    ∂FP2 =
        fixed_point_2_transfer(Val(Gauge), Val(Style), ∂A, A, A, X, X, O) +
        fixed_point_2_transfer(Val(Gauge), Val(Style), A, ∂A, A, X, X, O) +
        fixed_point_2_transfer(Val(Gauge), Val(Style), A, A, ∂A, X, X, O) +
        fixed_point_2_transfer(Val(Gauge), Val(Style), A, A, A, ∂X, X, O) +
        fixed_point_2_transfer(Val(Gauge), Val(Style), A, A, A, X, ∂X, O) -
        fixed_point_2_vector(Val(Gauge), Val(Style), ∂A, X, X, N) -
        fixed_point_2_vector(Val(Gauge), Val(Style), A, ∂X, X, N) -
        fixed_point_2_vector(Val(Gauge), Val(Style), A, X, ∂X, N) -
        fixed_point_2_vector(Val(Gauge), Val(Style), A, X, X, ∂N)
    return ∂FP2
end

## FP3: boundary normalization (isometry) condition

function fixed_point_3_pushforward(
    ::Val{:center},
    ::Val{Style},
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    ∂A::MPSKit.GenericMPSTensor{S},
    ∂X::MPSKit.MPSBondTensor{S},
) where {Style,S}
    ∂FP3 =
        fixed_point_3_transfer(Val(:center), Val(Style), ∂A, A, X, X) +
        fixed_point_3_transfer(Val(:center), Val(Style), A, ∂A, X, X) +
        fixed_point_3_transfer(Val(:center), Val(Style), A, A, ∂X, X) +
        fixed_point_3_transfer(Val(:center), Val(Style), A, A, X, ∂X) -
        fixed_point_3_vector(Val(:center), Val(Style), ∂X, X) -
        fixed_point_3_vector(Val(:center), Val(Style), X, ∂X)
    return ∂FP3
end
function fixed_point_3_pushforward(
    ::Val{:left},
    ::Val{:naive},
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    ∂A::MPSKit.GenericMPSTensor{S},
    ∂X::MPSKit.MPSBondTensor{S},
) where {S}
    ∂FP3 =
        fixed_point_3_transfer(Val(:left), Val(:naive), ∂A, A, X, X) +
        fixed_point_3_transfer(Val(:left), Val(:naive), A, ∂A, X, X)
    return ∂FP3
end
function fixed_point_3_pushforward(
    ::Val{:left},
    ::Val{:regularized},
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    ∂A::MPSKit.GenericMPSTensor{S},
    ∂X::MPSKit.MPSBondTensor{S},
) where {S}
    ∂FP3 =
        fixed_point_3_transfer(Val(:left), Val(:regularized), ∂A, A, X, X) +
        fixed_point_3_transfer(Val(:left), Val(:regularized), A, ∂A, X, X) +
        fixed_point_3_transfer(Val(:left), Val(:regularized), A, A, ∂X, X) +
        fixed_point_3_transfer(Val(:left), Val(:regularized), A, A, X, ∂X) -
        fixed_point_3_vector(Val(:left), Val(:regularized), ∂X, X) -
        fixed_point_3_vector(Val(:left), Val(:regularized), X, ∂X)
    return ∂FP3
end

## FP4: corner normalization normalization condition

function fixed_point_4_pushforward(
    ::Val{Gauge}, ::Val{Style}, X::MPSKit.MPSBondTensor{S}, ∂X::MPSKit.MPSBondTensor{S}
) where {Gauge,Style,S}
    return 4 * tr(X^3 * ∂X)
end

# partial pushforward implementing environment JVP

# TODO: figure out if we ever actually need to do this
function project_hermitian(A::MPSKit.GenericMPSTensor)
    A´ = (A + physical_flip(_conj(A))) / 2
    return A´
end

function generate_partial_pushforward(
    ::Val{Gauge},
    ::Val{Style},
    network::InfiniteSquareNetwork,
    A::MPSKit.GenericMPSTensor{S},
    X::MPSKit.MPSBondTensor{S},
    N::Number,
) where {Gauge,Style,S}
    O = network[1, 1]

    function partial_pushforward((∂A, ∂X, ∂N))
        # ∂A = project_hermitian(∂A)

        # Hermiticity
        ∂FP1 = fixed_point_1_pushforward(Val(Gauge), Val(Style), A, X, ∂A, ∂X)

        # Eigenvalue equation
        ∂FP2 = fixed_point_2_pushforward(Val(Gauge), Val(Style), A, X, N, O, ∂A, ∂X, ∂N)

        # Left fixed point condition
        ∂FP3 = fixed_point_3_pushforward(Val(Gauge), Val(Style), A, X, ∂A, ∂X)

        # Normalization
        ∂FP4 = fixed_point_4_pushforward(Val(Gauge), Val(Style), X, ∂X)

        return (∂FP1, ∂FP2, ∂FP3, ∂FP4)
        # return (∂FP2, ∂FP3, ∂FP4)
    end

    return partial_pushforward
end

#
# Pulling through pullback contractions
#

# TODO: implement manual pullback for all combinations, which is a total pain...

# partial pullback implementing environment VJP

# A pullback contractions
function FP2_O_open_north(
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
    ΔFP2::MPSKit.GenericMPSTensor{S,3},
) where {S}
    AWU = physical_flip(A)
    ASU = physical_flip(A)
    ΔFP2U = physical_flip(ΔFP2)
    @autoopt @tensor ΔA[χ_NNW D_N_above D_N_below; χ_NE] :=
        conj(AWU[χ_WSW D_W_above D_W_below; χ_WNW]) *
        conj(ASU[χ_SE D_S_above D_S_below; χ_SSW]) *
        conj(X[χ_WNW; χ_NNW]) *
        conj(X[χ_SSW; χ_WSW]) *
        conj(ket(O)[d; D_N_above D_E_above D_S_above D_W_above]) *
        bra(O)[d; D_N_below D_E_below D_S_below D_W_below] *
        ΔFP2U[χ_SE D_E_above D_E_below; χ_NE]
    return ΔA
end

function FP2_O_open_west(
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
    ΔFP2::MPSKit.GenericMPSTensor{S,3},
) where {S}
    AN = A
    ASU = physical_flip(A)
    ΔFP2U = physical_flip(ΔFP2)
    @autoopt @tensor ΔA[χ_WSW D_W_above D_W_below; χ_WNW] :=
        conj(AN[χ_NNW D_N_above D_N_below; χ_NE]) *
        conj(ASU[χ_SE D_S_above D_S_below; χ_SSW]) *
        conj(X[χ_WNW; χ_NNW]) *
        conj(X[χ_SSW; χ_WSW]) *
        conj(ket(O)[d; D_N_above D_E_above D_S_above D_W_above]) *
        bra(O)[d; D_N_below D_E_below D_S_below D_W_below] *
        ΔFP2U[χ_SE D_E_above D_E_below; χ_NE]
    return physical_flip(ΔA)
end

function FP2_O_open_south(
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
    ΔFP2::MPSKit.GenericMPSTensor{S,3},
) where {S}
    AN = A
    AWU = physical_flip(A)
    ΔFP2U = physical_flip(ΔFP2)
    @autoopt @tensor ΔA[χ_SE D_S_above D_S_below; χ_SSW] :=
        conj(AN[χ_NNW D_N_above D_N_below; χ_NE]) *
        conj(AWU[χ_WSW D_W_above D_W_below; χ_WNW]) *
        conj(X[χ_WNW; χ_NNW]) *
        conj(X[χ_SSW; χ_WSW]) *
        conj(ket(O)[d; D_N_above D_E_above D_S_above D_W_above]) *
        bra(O)[d; D_N_below D_E_below D_S_below D_W_below] *
        ΔFP2U[χ_SE D_E_above D_E_below; χ_NE]
    return physical_flip(ΔA)
end

function FP2_N_open_west(
    X::MPSKit.MPSBondTensor{S}, N::Number, ΔFP2::MPSKit.GenericMPSTensor{S,3}
) where {S}
    @autoopt @tensor ΔA[χ_SW D_above D_below; χ_NW] :=
        conj(X[χ_NW; χ_NE]) * conj(X[χ_SE; χ_SW]) * ΔFP2[χ_SE D_above D_below; χ_NE]
    return N * ΔA
end

function FP3_open_north(
    A::MPSKit.GenericMPSTensor{S,3},
    X2::MPSKit.MPSBondTensor{S},
    ΔFP3::MPSKit.MPSBondTensor{S},
) where {S}
    return @autoopt @tensor ΔA[χ_NW D_above D_below; χ_NE] :=
        conj(X2[χ_SW; χ_NW]) * A[χ_SW D_above D_below; χ_SE] * ΔFP3[χ_SE; χ_NE]
end

function FP3_open_south(
    A::MPSKit.GenericMPSTensor{S,3},
    X2::MPSKit.MPSBondTensor{S},
    ΔFP3::MPSKit.MPSBondTensor{S},
) where {S}
    return @autoopt @tensor ΔA[χ_SW D_above D_below; χ_SE] :=
        X2[χ_SW; χ_NW] * A[χ_NW D_above D_below; χ_NE] * conj(ΔFP3[χ_SE; χ_NE])
end

# X pullback contractions
function FP2_O_open_northwest(
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
    ΔFP2::MPSKit.GenericMPSTensor{S,3},
) where {S}
    AWU = physical_flip(A)
    ASU = physical_flip(A)
    ΔFP2U = physical_flip(ΔFP2)
    return @autoopt @tensor ΔX[χ_WNW; χ_NNW] :=
        conj(A[χ_NNW D_N_above D_N_below; χ_NE]) *
        conj(AWU[χ_WSW D_W_above D_W_below; χ_WNW]) *
        conj(ASU[χ_SE D_S_above D_S_below; χ_SSW]) *
        conj(X[χ_SSW; χ_WSW]) *
        conj(ket(O)[d; D_N_above D_E_above D_S_above D_W_above]) *
        bra(O)[d; D_N_below D_E_below D_S_below D_W_below] *
        ΔFP2U[χ_SE D_E_above D_E_below; χ_NE]
end
function FP2_O_open_southwest(
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
    ΔFP2::MPSKit.GenericMPSTensor{S,3},
) where {S}
    AWU = physical_flip(A)
    ASU = physical_flip(A)
    ΔFP2U = physical_flip(ΔFP2)
    return @autoopt @tensor ΔX[χ_SSW; χ_WSW] :=
        conj(A[χ_NNW D_N_above D_N_below; χ_NE]) *
        conj(AWU[χ_WSW D_W_above D_W_below; χ_WNW]) *
        conj(ASU[χ_SE D_S_above D_S_below; χ_SSW]) *
        conj(X[χ_WNW; χ_NNW]) *
        conj(ket(O)[d; D_N_above D_E_above D_S_above D_W_above]) *
        bra(O)[d; D_N_below D_E_below D_S_below D_W_below] *
        ΔFP2U[χ_SE D_E_above D_E_below; χ_NE]
end
function FP2_N_open_north(
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    N::Number,
    ΔFP2::MPSKit.GenericMPSTensor{S,3},
) where {S}
    @autoopt @tensor ΔX[χ_NW; χ_NE] :=
        conj(A[χ_SW D_above D_below; χ_NW]) *
        conj(X[χ_SE; χ_SW]) *
        ΔFP2[χ_SE D_above D_below; χ_NE]
    return N * ΔX
end
function FP2_N_open_south(
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    N::Number,
    ΔFP2::MPSKit.GenericMPSTensor{S,3},
) where {S}
    @autoopt @tensor ΔX[χ_SE; χ_SW] :=
        conj(A[χ_SW D_above D_below; χ_NW]) *
        conj(X[χ_NW; χ_NE]) *
        ΔFP2[χ_SE D_above D_below; χ_NE]
    return N * ΔX
end

function FP3_open_northwest(
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    ΔFP3::MPSKit.MPSBondTensor{S},
) where {S}
    return @autoopt @tensor ΔX[χ_WNW; χ_NNW] :=
        X[χ_WNW; χ_SW] *
        conj(A[χ_NNW D_above D_below; χ_NE]) *
        A[χ_SW D_above D_below; χ_SE] *
        ΔFP3[χ_SE; χ_NE]
end

function FP3_open_southwest(
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    ΔFP3::MPSKit.MPSBondTensor{S},
) where {S}
    return @autoopt @tensor ΔX[χ_WSW; χ_SSW] :=
        X[χ_WSW; χ_NW] *
        A[χ_NW D_above D_below; χ_NE] *
        conj(A[χ_SSW D_above D_below; χ_SE]) *
        conj(ΔFP3[χ_SE; χ_NE])
end

function FP3_open_north(X::MPSKit.MPSBondTensor{S}, ΔFP3::MPSKit.MPSBondTensor{S}) where {S}
    return @autoopt @tensor ΔX[χ_NW; χ_NE] := X[χ_NW; χ_S] * ΔFP3[χ_S; χ_NE]
end
function FP3_open_south(X::MPSKit.MPSBondTensor{S}, ΔFP3::MPSKit.MPSBondTensor{S}) where {S}
    return @autoopt @tensor ΔX[χ_SW; χ_SE] := X[χ_SW; χ_N] * conj(ΔFP3[χ_SE; χ_N])
end

function FP4(X::MPSKit.MPSBondTensor{S}, ΔFP4::Number) where {S}
    return 4 * ΔFP4 * (X')^3
end

# put everything together
function generate_partial_pullback(::Val{:center}, ::Val{:naive}, network, A, X, N)
    O = network[1, 1]

    function partial_pullback((ΔFP1, ΔFP2, ΔFP3, ΔFP4))
        # ΔA

        # FP1
        ΔA1 = ΔFP1 - physical_flip(_conj(ΔFP1))

        # FP2
        ΔA2 =
            FP2_O_open_north(A, X, O, ΔFP2) +
            FP2_O_open_west(A, X, O, ΔFP2) +
            FP2_O_open_south(A, X, O, ΔFP2) - FP2_N_open_west(X, N, ΔFP2)

        # FP3
        ΔA3 = FP3_open_north(A, X' * X, ΔFP3) + FP3_open_south(A, X' * X, ΔFP3)

        ΔA = ΔA1 + ΔA2 + ΔA3

        # ΔX

        # FP2
        ΔX2 =
            FP2_O_open_northwest(A, X, O, ΔFP2) + FP2_O_open_southwest(A, X, O, ΔFP2) -
            FP2_N_open_north(A, X, N, ΔFP2) - FP2_N_open_south(A, X, N, ΔFP2)

        # FP3
        ΔX3 =
            FP3_open_northwest(A, X, ΔFP3) +
            FP3_open_southwest(A, X, ΔFP3) +
            FP3_open_north(X, ΔFP3) +
            FP3_open_south(X, ΔFP3)

        # FP4
        ΔX4 = FP4(X, ΔFP4)

        ΔX = ΔX2 + ΔX3 + ΔX4

        # ΔN
        ΔN =
            -@autoopt @tensor conj(A[χ_SW D_above D_below; χ_NW]) *
                conj(X[χ_NW; χ_NE]) *
                conj(X[χ_SE; χ_SW]) *
                ΔFP2[χ_SE D_above D_below; χ_NE]

        return (ΔA, ΔX, ΔN)
    end

    return partial_pullback
end
