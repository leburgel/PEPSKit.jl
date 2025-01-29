"""
Simple symmetrization for a single site unit cell.
"""

using PEPSKit

struct MyRotate <: PEPSKit.SymmetrizationStyle end
struct MyRotateReflect <: PEPSKit.SymmetrizationStyle end

function PEPSKit.symmetrize!(peps::InfinitePEPS, ::MyRotate)
    @assert length(peps) == 1 "MyRotate only works for single site unit cells"
    peps[1] = PEPSKit.rot_inv(peps[1])
    return peps
end

function PEPSKit.symmetrize!(peps::InfinitePEPS, ::MyRotateReflect)
    @assert length(peps) == 1 "MyRotateReflect only works for single site unit cells"
    peps[1] = PEPSKit.herm_depth_inv(PEPSKit.rot_inv(peps[1]))
    return peps
end

function check_symmetry(peps::InfinitePEPS, ::MyRotate; tol=1e-10)
    println("Symmetrizing with MyRotate...")
    @assert length(peps) == 1 "check_symmetry only works for single site unit cells"
    @assert norm(peps[1] - PEPSKit.rotl90(peps[1])) / norm(peps[1]) < tol
end

function check_symmetry(peps::InfinitePEPS, ::MyRotateReflect; tol=1e-10)
    println("Symmetrizing with MyRotateReflect...")
    @assert length(peps) == 1 "check_symmetry only works for single site unit cells"
    @assert norm(peps[1] - PEPSKit._fit_spaces(PEPSKit.rotl90(peps[1]), peps[1])) /
            norm(peps[1]) < tol "not rotation invariant"
    @assert norm(peps[1] - PEPSKit._fit_spaces(PEPSKit.herm_depth(peps[1]), peps[1])) /
            norm(peps[1]) < tol "not hermitian-reflection invariant"
end
