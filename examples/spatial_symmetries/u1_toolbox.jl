using TensorKit
using PEPSKit
using ChainRulesCore

# extend spatial symmetry toolbox for a specific U1-symmetric use case

struct U1HReflection <: SymmetrizationStyle end
struct U1HReflectionRotation <: SymmetrizationStyle end
struct U1XHReflection <: SymmetrizationStyle end
struct U1XHReflectionRotation <: SymmetrizationStyle end

struct U1Symmetric <: UnitCellStyle end # trivial flipper
struct U1XSymmetric <: UnitCellStyle end # non-trivial flipper

## U1 charge conjugation

# no actual duals, since this does NOT flip any arrows
charge_conj(s::GradedSpace) = flip(s')
function charge_conj(f::FusionTree)
    return FusionTree(conj.(f.uncoupled), conj(f.coupled), f.isdual)
end
function charge_conj(t::TensorMap)
    codom = ProductSpace(prod(charge_conj(s) for s in codomain(t)))
    dom = prod(charge_conj(s) for s in domain(t))

    toret = TensorMap(zeros, ComplexF64, codom ← dom)
    for (s, f) in fusiontrees(t)
        toret[charge_conj(s), charge_conj(f)] = t[s, f]
    end

    return toret
end
function ChainRulesCore.rrule(::typeof(charge_conj), t::AbstractTensorMap)
    t´ = charge_conj(t)
    function charge_conj_pullback(Δt)
        ∂t = charge_conj(Δt)
        return NoTangent(), ∂t
    end
    return t´, charge_conj_pullback
end

## U1 symmetric unit cell

function u1_flipper(A::PEPSTensor{U1Space}, i::Int)
    I = isomorphism(flip(space(A, i)), space(A, i))
    X = -1 * I # all blocks with nontrivial charge get a minus sign
    block(X, U1Irrep(0)) .*= -1 # but zero charge block is still just the identity
    return X
end
@non_differentiable u1_flipper(args...)

# hack into `fill_peps` specifically for this use case
function fill_peps(A::PEPSTensor, ::U1Symmetric)
    Ac = charge_conj(A)
    @tensor B[-1; -2 -3 -4 -5] :=
        Ac[-1; 1 2 3 4] *
        flipper(Ac, 2)[-2; 1] *
        flipper(Ac, 3)[-3; 2] *
        flipper(Ac, 4)[-4; 3] *
        flipper(Ac, 5)[-5; 4]
    return InfinitePEPS([A B; B A])
end
function fill_peps(A::PEPSTensor{U1Space}, ::U1XSymmetric)
    Ac = charge_conj(A)
    @tensor B[-1; -2 -3 -4 -5] :=
        Ac[-1; 1 2 3 4] *
        u1_flipper(Ac, 2)[-2; 1] *
        u1_flipper(Ac, 3)[-3; 2] *
        u1_flipper(Ac, 4)[-4; 3] *
        u1_flipper(Ac, 5)[-5; 4]
    return InfinitePEPS([A B; B A])
end

## custom U1 spatial symmetry conditions for this use case

function u1_spaceflip(A::PEPSTensor{U1Space})
    @tensor A´[-1; -2 -3 -4 -5] :=
        A[1; 2 3 4 5] *
        u1_flipper(A, 1)[-1; 1] *
        u1_flipper(A, 2)[-2; 2] *
        u1_flipper(A, 3)[-3; 3] *
        u1_flipper(A, 4)[-4; 4] *
        u1_flipper(A, 5)[-5; 5]
    return A´
end

# regular flipper
u1_hvflip_inv(A::PEPSTensor) = 0.5 * (A + hvflip(spaceflip(charge_conj(A))))
u1_hhflip_inv(A::PEPSTensor) = 0.5 * (A + hvflip(spaceflip(charge_conj(A))))
u1_hvflip_rot_inv(A::PEPSTensor) = u1_hvflip_inv(rot_inv(A))

symmetrize(A::PEPSTensor, ::U1HReflection) = u1_hvflip_inv(A) # only impose 'vertical' hermiticity for now
symmetrize(A::PEPSTensor, ::U1HReflectionRotation) = u1_hvflip_rot_inv(A) # only impose 'vertical' hermiticity for now

# specifal flipper
u1_xhvflip_inv(A::PEPSTensor{U1Space}) = 0.5 * (A + hvflip(u1_spaceflip(charge_conj(A))))
u1_xhhflip_inv(A::PEPSTensor{U1Space}) = 0.5 * (A + hvflip(u1_spaceflip(charge_conj(A))))
u1_xhvflip_rot_inv(A::PEPSTensor{U1Space}) = u1_xhvflip_inv(rot_inv(A))

symmetrize(A::PEPSTensor{U1Space}, ::U1XHReflection) = u1_xhvflip_inv(A) # only impose 'vertical' hermiticity for now
symmetrize(A::PEPSTensor{U1Space}, ::U1XHReflectionRotation) = u1_xhvflip_rot_inv(A) # only impose 'vertical' hermiticity for now
