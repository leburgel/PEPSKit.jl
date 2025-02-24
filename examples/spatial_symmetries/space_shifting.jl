using TensorKit
using PEPSKit
import MPSKit: tensorexpr, PeriodicArray, add_physical_charge

@generated function _fuse_isomorphisms(
    op::AbstractTensorMap{<:Any,S,N,N}, fs::Vector{<:AbstractTensorMap{<:Any,S,1,2}}
) where {S,N}
    op_out_e = tensorexpr(:op_out, -(1:N), -((1:N) .+ N))
    op_e = tensorexpr(:op, 1:3:(3 * N), 2:3:(3 * N))
    f_es = map(1:N) do i
        j = 3 * (i - 1) + 1
        return tensorexpr(:(fs[$i]), -i, (j, j + 2))
    end
    f_dag_es = map(1:N) do i
        j = 3 * (i - 1) + 1
        return tensorexpr(:(fs[$i]), -(N + i), (j + 1, j + 2))
    end
    multiplication_ex = Expr(
        :call, :*, op_e, f_es..., map(x -> Expr(:call, :conj, x), f_dag_es)...
    )
    return macroexpand(@__MODULE__, :(return @tensor $op_out_e := $multiplication_ex))
end

"""
Fuse identities on auxiliary physical spaces into a given operator.
"""
function _fuse_ids(op::AbstractTensorMap{T,S,N,N}, Ps::NTuple{N,S}) where {T,S,N}
    # make isomorphisms
    fs = map(1:N) do i
        return isomorphism(fuse(space(op, i), Ps[i]), space(op, i) ⊗ Ps[i])
    end
    # and fuse them into the operator
    return _fuse_isomorphisms(op, fs)
end

# TODO: move all of this into PEPSKit
TensorKit.sectortype(O::LocalOperator) = sectortype(typeof(O))
TensorKit.sectortype(::Type{<:LocalOperator{T,S}}) where {T,S} = sectortype(S)
"""
    add_physical_charge(H::LocalOperator, charges::AbstractMatrix{<:Sector}) where {S}

Change the spaces of a `LocalOperator` by fusing in an auxiliary charge on every site,
according to a given matrix of 'auxiliary' physical charges.
"""
function add_physical_charge(H::LocalOperator, charges::AbstractMatrix{<:Sector})
    size(H.lattice) == size(charges) ||
        throw(ArgumentError("Incompatible lattice and auxiliary charge sizes"))
    sectortype(H) === eltype(charges) ||
        throw(SectorMismatch("Incompatible lattice and auxiliary charge sizes"))

    Paux = PeriodicArray(map(charge -> Vect[typeof(charge)](charge => 1), charges))

    # new physical spaces
    Pspaces = map(fuse, H.lattice, Paux)

    new_terms = map(H.terms) do (sites, op)
        Paux_slice = map(Base.Fix1(getindex, Paux), sites)
        return sites => _fuse_ids(op, Paux_slice)
    end
    H´ = LocalOperator(Pspaces, new_terms...)

    return H´
end
