# some plumbing for a fair error comparison between different algorithms

using TensorKit
using MPSKit
using PEPSKit

# state tracking

Base.@kwdef mutable struct StateTracker
    state = nothing
end

function get_error!(env, tracker::StateTracker)
    if isnothing(tracker.state)
        ϵ = 1.0
    else
        ϵ, = PEPSKit.calc_convergence(env, tracker.state)
    end
    tracker.state = env
    return ϵ
end

# implement singular value distance for all environment types

function PEPSKit.calc_convergence(
    envs_new::Union{InfiniteMPS,MultilineMPS}, envs_old::Union{InfiniteMPS,MultilineMPS}
)
    CS_new = map(x -> tsvd(x)[2], envs_new.C)
    CS_old = map(x -> tsvd(x)[2], envs_old.C)
    return maximum(PEPSKit._singular_value_distance, zip(CS_old, CS_new))
end

function PEPSKit.calc_convergence(envs_new::PullingThroughEnv, envs_old::PullingThroughEnv)
    (isnothing(envs_old.LN) || isnothing(envs_old.RW)) && return one(scalartype(envs_new.N))
    L_new = tsvd(envs_new.LN)[2]
    L_old = tsvd(envs_old.LN)[2]
    R_new = tsvd(envs_new.RW)[2]
    R_old = tsvd(envs_old.RW)[2]
    return maximum(PEPSKit._singular_value_distance, zip([L_old, R_old], [L_new, R_new]))
end

# adapt to different finalize signatures

function ctm_error_tracker(errs::Vector{Float64}, envinit=nothing)
    tracker = StateTracker(envinit)
    function finalize(iter, env, state)
        ϵ = get_error!(env, tracker)
        push!(errs, ϵ)
        return env
    end
    return finalize
end

function vumps_error_tracker(errs::Vector{Float64}, envinit=nothing)
    tracker = StateTracker(envinit)
    function finalize(iter, ψ, H, envs)
        ϵ = get_error!(ψ, tracker)
        push!(errs, ϵ)
        return ψ, envs
    end
    return finalize
end

const pt_error_tracker = ctm_error_tracker # same finalize signature
