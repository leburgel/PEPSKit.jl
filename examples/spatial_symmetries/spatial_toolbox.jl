import Pkg
Pkg.add(url="https://github.com/tangwei94/SpatiallySymmetricTensor.jl", rev="main")

using TensorKit
using PEPSKit
using PEPSKit: PEPSTensor
using SpatiallySymmetricTensor
using Zygote

# Style definitions
# -----------------

# symmetrization styles
abstract type SymmetrizationStyle end
struct None <: SymmetrizationStyle end
struct Rotation <: SymmetrizationStyle end # C4
struct Reflection <: SymmetrizationStyle end # D2
struct ReflectionRotation <: SymmetrizationStyle end # C4v
struct HReflection <: SymmetrizationStyle end # D2
struct HReflectionRotation <: SymmetrizationStyle end # C4v 

# unit cell styles
abstract type UnitCellStyle end
struct Asymmetric <: UnitCellStyle end # 1x1 unit cell
struct Symmetric <: UnitCellStyle end # 2x2 unit cell

# Filling up a PEPS unit cell
# ---------------------------
flipper(A::PEPSTensor, i::Int) = isomorphism(flip(space(A, i)), space(A, i))

# fill up InfinitePEPS unit cell from a single PEPSTensor
function fill_peps(A::PEPSTensor, ::Symmetric)
    @tensor B[-1; -2 -3 -4 -5] :=
        A[-1; 1 2 3 4] *
        flipper(A, 2)[-2; 1] *
        flipper(A, 3)[-3; 2] *
        flipper(A, 4)[-4; 3] *
        flipper(A, 5)[-5; 4]
    return InfinitePEPS([A B; B A])
end
function fill_peps(A::PEPSTensor, ::Asymmetric)
    @tensor A´[-1; -2 -3 -4 -5] :=
        A[-1; 1 2 -4 -5] * flipper(A, 2)[-2; 1] * flipper(A, 3)[-3; 2]
    return InfinitePEPS(A´)
end

# Symmetry operations on PEPS tensors
# -----------------------------------

# rotations: defined in PEPSKit.jl
# rotl90 (counterclockwise):    ((1,), (3, 4, 5, 2))
# rotr90 (clockwise):           ((1,), (5, 2, 3, 4))
# rot180:                       ((1,), (4, 5, 2, 3))

# reflections:
vflip(A::PEPSTensor) = permute(A, ((1,), (4, 3, 2, 5)))
hflip(A::PEPSTensor) = permute(A, ((1,), (4, 3, 2, 5)))
dflip1(A::PEPSTensor) = permute(A, ((1,), (3, 2, 5, 4)))
dflip2(A::PEPSTensor) = permute(A, ((1,), (5, 4, 3, 2)))

# hermitian reflections:
hvflip(A::PEPSTensor) = permute(A', ((5,), (3, 2, 1, 4)))
hhflip(A::PEPSTensor) = permute(A', ((5,), (1, 4, 3, 2)))

# impose regular spatial symmetries
rot_inv(A::PEPSTensor) = 0.25 * (A + rotl90(A) + rotr90(A) + rot180(A))

vflip_inv(A::PEPSTensor) = 0.5 * (A + vflip(A))
hflip_inv(A::PEPSTensor) = 0.5 * (A + hflip(A))

flip_inv(A::PEPSTensor) = hflip_inv(vflip_inv((A)))

flip_rot_inv(A::PEPSTensor) = flip_inv(rot_inv(A))

# impose hermitian flip symmetries; requires flippers due to presence of adjoint
function spaceflip(A::PEPSTensor)
    @tensor A´[-1; -2 -3 -4 -5] :=
        A[1; 2 3 4 5] *
        flipper(A, 1)[-1; 1] *
        flipper(A, 2)[-2; 2] *
        flipper(A, 3)[-3; 3] *
        flipper(A, 4)[-4; 4] *
        flipper(A, 5)[-5; 5]
    return A´
end
hvflip_inv(A::PEPSTensor) = 0.5 * (A + spaceflip(hvflip(A)))
hhflip_inv(A::PEPSTensor) = 0.5 * (A + spaceflip(hhflip(A)))

hflip_inv(A::PEPSTensor) = hvflip_inv(hhflip_inv(A))

hflip_rot_inv(A::PEPSTensor) = hvflip_inv(hhflip_inv(rot_inv(A)))

# define function for manually imposing the spatial symmetries
symmetrize(A::PEPSTensor, ::None) = A
symmetrize(A::PEPSTensor, ::Rotation) = rot_inv(A)
symmetrize(A::PEPSTensor, ::Reflection) = flip_inv(A)
symmetrize(A::PEPSTensor, ::ReflectionRotation) = flip_rot_inv(A)
symmetrize(A::PEPSTensor, ::HReflection) = hflip_inv(A)
symmetrize(A::PEPSTensor, ::HReflectionRotation) = hflip_rot_inv(A)

# Manifestly spatially symmetric tensors
# --------------------------------------

# map symmetrization style to point group and corresponding representations for real and
# imaginary parts
get_point_group(::SymmetrizationStyle) = nothing, nothing
get_point_group(::Rotation) = C4(), (:A, :A)
get_point_group(::Reflection) = D2(), (:A, :A)
get_point_group(::ReflectionRotation) = C4v(), (:A1, :A1)
get_point_group(::HReflection) = D2(), (:A, :B1)
get_point_group(::HReflectionRotation) = C4v(), (:A1, :A2)

function find_symmetric_basis(
    P::S, V::S, symm_style::SymmetrizationStyle
) where {S<:ElementarySpace}
    A0 = TensorMap(zeros, ComplexF64, P ← V ⊗ V ⊗ V ⊗ V)
    point_group, (real_rep, imag_rep) = get_point_group(symm_style)
    A_real = find_solution(point_group, A0, real_rep)
    A_imag = find_solution(point_group, A0, imag_rep)
    A_basis = vcat(A_real, 1im .* A_imag)
    return A_basis
end

vec2peps(a::Vector{<:Real}, A_basis::Vector{<:PEPSTensor}) = sum(a .* A_basis)

# PEPS optimization with manual symmetrization
# --------------------------------------------

function peps_opt_costfunction(;
    boundary_alg=CTMRG(),
    gradient_alg=GMRES(),
    reuse_env=true,
    unitcell_style=Asymmetric(),
    symm_style=HReflRotation(),
)
    function peps_cfun(x)
        (A::PEPSTensor, env::CTMRGEnv) = x
        E, g = withgradient(A) do x
            ψ = fill_peps(x, unitcell_style) # the first bamboozle
            env´ = PEPSKit.hook_pullback(
                leading_boundary, env, ψ, boundary_alg; alg_rrule=gradient_alg
            )
            ignore_derivatives() do
                reuse_env && PEPSKit.update!(env, env´) # in-place update CTMRG environments
            end
            return costfun(ψ, env´, H)
        end
        g = symmetrize(only(g), symm_style) # the second bamboozle

        return E, g
    end

    function peps_retract(x, η, α)
        A = deepcopy(x[1])
        A += η * α
        env = deepcopy(x[2])
        A = symmetrize(A, symm_style)
        return (A, env), η
    end

    peps_inner(_, η₁, η₂) = real(dot(η₁, η₂))

    return peps_cfun, peps_retract, peps_inner
end

# Vector optimization with manifest symmetries
# --------------------------------------------

function vector_opt_costfunction(
    P::S,
    V::S;
    boundary_alg=CTMRG(),
    gradient_alg=GMRES(),
    reuse_env=true,
    unitcell_style=Asymmetric(),
    symm_style=HReflRotation(),
) where {S<:ElementarySpace}
    A_basis = find_symmetric_basis(P, V, symm_style)

    function vector_cfun(x)
        (a::Vector{<:Real}, env::CTMRGEnv) = x
        E, g = withgradient(a) do v
            ψ = fill_peps(vec2peps(v, A_basis), unitcell_style) # the bamboozle
            env´ = PEPSKit.hook_pullback(
                leading_boundary, env, ψ, boundary_alg; alg_rrule=gradient_alg
            )
            ignore_derivatives() do
                reuse_env && PEPSKit.update!(env, env´) # in-place update CTMRG environments
            end
            return costfun(ψ, env´, H)
        end

        return E, only(g)
    end

    function vector_retract(x, η, α)
        a = deepcopy(x[1])
        a += η * α
        env = deepcopy(x[2])
        return (a, env), η
    end

    vector_inner(_, η₁, η₂) = dot(η₁, η₂)

    return vector_cfun, vector_retract, vector_inner
end
