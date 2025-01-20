
# PEPS contractions

function transfer_north(
    N::MPSKit.GenericMPSTensor{S,3},
    WR::MPSKit.GenericMPSTensor{S,3},
    ket::PEPSTensor,
    bra::PEPSTensor,
) where {S}
    return @autoopt @tensor N´[χ_SW D_S_above D_S_below; χ_SE] :=
        WR[χ_SW D_W_above D_W_below; χ_NW] *
        N[χ_NW D_N_above D_N_below; χ_NE] *
        ket[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra[d; D_N_below D_E_below D_S_below D_W_below]) *
        conj(WR[χ_SE, D_E_above, D_E_below; χ_NE])
end

function transfer_west(
    W::MPSKit.GenericMPSTensor{S,3},
    NL::MPSKit.GenericMPSTensor{S,3},
    ket::PEPSTensor,
    bra::PEPSTensor,
) where {S}
    return @autoopt @tensor W´[χ_SE D_E_above D_E_below; χ_NE] :=
        NL[χ_NW D_N_above D_N_below; χ_NE] *
        W[χ_SW D_W_above D_W_below; χ_NW] *
        ket[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra[d; D_N_below D_E_below D_S_below D_W_below]) *
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
