# Generic routine to fuse auxiliary physical spaces into a PEPS and a Hamiltonian,
# essentially 'shifting' the physical charges in a consistent way.
# Just so we don't have to have dedicated types to deal with auxiliary legs... 

import MPSKit: tensorexpr

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

"""
    shift_physical_spaces(H::LocalOperator, Paux::Matrix{S}) where {S}

Shift spaces of a `LocalOperator` according to a given matrix of 'auxiliary' physical
spaces.
"""
function shift_physical_spaces(H::LocalOperator{T,S}, Paux::Matrix{S}) where {T,S}
    @assert size(H.lattice) == size(Paux) "Incompatible lattice and auxiliary space sizes"
    # new physical spaces
    Pspaces = map(fuse, H.lattice, Paux)
    # make auxiliary space indexing periodic
    Paux = PeriodicArray(Paux)

    new_terms = map(H.terms) do (sites, op)
        Paux_slice = map(Base.Fix1(getindex, Paux), sites)
        return sites => _fuse_ids(op, Paux_slice)
    end
    H´ = LocalOperator(Pspaces, new_terms...)

    return H´, Pspaces
end
