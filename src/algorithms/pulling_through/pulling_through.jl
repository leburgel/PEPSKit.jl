"""
    PullingThrough

Pulling-through contraction algorithm.
"""
@kwdef struct PullingThrough
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity

    alg_gauge = MPSKit.Defaults.alg_gauge(; verbosity=1, maxiter=100)
    alg_eigsolve = MPSKit.Defaults.alg_eigsolve(; ishermitian=false)
end

function normalize_ish(A::MPSKit.GenericMPSTensor; tol=1e-12)
    init = MPSKit.randomize!(similar(A, space(A, 1), space(A, 1)))
    vals, = eigsolve(flip(MPSKit.TransferMatrix(A, A)),
                                              init, 1, :LM; tol=tol)
    λ = first(vals)
    return A / sqrt(abs(first(λ)))
end

#
# Iterative contraction routine
#

"""
    pulling_through_update(state, env, alg::PullingThrough) -> env′, info

Perform a single pulling through iteration.
"""
function pulling_through_update(
    state::InfinitePEPS, env::PullingThroughEnv, alg_eigsolve, alg_gauge
)
    # update north
    env = gauge_west(env, alg_gauge)
    function _tn(x)
        return transfer_north(x, env.WR, only(state.A), only(state.A))
    end
    λ, N_next = MPSKit.fixedpoint(_tn, env.N, :LM, alg_eigsolve)
    @reset env.N = N_next
    # @reset env.N = normalize_ish(N_next) # TODO: normalize? how important is this?

    # update west
    env = gauge_north(env, alg_gauge)
    function _tw(x)
        return transfer_west(x, env.NL, only(state.A), only(state.A))
    end
    λ, W_next = MPSKit.fixedpoint(_tw, env.W, :LM, alg_eigsolve)
    @reset env.W = W_next
    # @reset env.W = normalize_ish(W_next) # TODO: normalize? how important is this?
    
    return env, λ
end
function pulling_through_update(
    partfunc::InfinitePartitionFunction, env::PullingThroughEnv, alg_eigsolve, alg_gauge
)
    # update north
    env = gauge_west(env, alg_gauge)
    function _tn(x)
        return transfer_north(x, env.WR, only(partfunc.A))
    end
    λ, N_next = MPSKit.fixedpoint(_tn, env.N, :LM, alg_eigsolve)

    @reset env.N = N_next
    # @reset env.N = normalize_ish(N_next) # TODO: normalize? how important is this?

    # update west
    env = gauge_north(env, alg_gauge)
    function _tw(x)
        return transfer_west(x, env.NL, only(partfunc.A))
    end
    _, W_next = MPSKit.fixedpoint(_tw, env.W, :LM, alg_eigsolve)
    @reset env.W = W_next
    # @reset env.W = normalize_ish(W_next) # TODO: normalize? how important is this?

    return env, λ
end

"""
    pulling_through_iterate(env, state, alg::PullingThrough)

Converge a northwest pulling through corner for a given state.
"""
function pulling_through_iterate(envinit, state, alg::PullingThrough)
    ϵ::Float64 = calc_convergence(envinit)
    N = 0.0 # TODO: get norm from intermediate pulling through thing
    env = deepcopy(envinit)
    log = ignore_derivatives(() -> MPSKit.IterLog("PT"))

    LoggingExtras.withlevel(; alg.verbosity) do
        pt_loginit!(log, ϵ, N)
        for iter in 1:(alg.maxiter)
            alg_eigsolve = MPSKit.updatetol(alg.alg_eigsolve, iter, ϵ)
            alg_gauge = MPSKit.updatetol(alg.alg_gauge, iter, ϵ)

            env, N = pulling_through_update(state, env, alg_eigsolve, alg_gauge)

            ϵ = calc_convergence(env)

            if ϵ <= alg.tol
                pt_logfinish!(log, iter, ϵ, N)
                break
            end
            if iter == alg.maxiter
                pt_logcancel!(log, iter, ϵ, N)
            else
                pt_logiter!(log, iter, ϵ, N)
            end
        end
    end

    return env, N, ϵ
end

# custom pulling through logging
pt_loginit!(log, ϵ, N) = @infov 2 loginit!(log, ϵ, N)
pt_logiter!(log, iter, ϵ, N) = @infov 3 logiter!(log, iter, ϵ, N)
pt_logfinish!(log, iter, ϵ, N) = @infov 2 logfinish!(log, iter, ϵ, N)
pt_logcancel!(log, iter, ϵ, N) = @warnv 1 logcancel!(log, iter, ϵ, N)

@non_differentiable pt_loginit!(args...)
@non_differentiable pt_logiter!(args...)
@non_differentiable pt_logfinish!(args...)
@non_differentiable pt_logcancel!(args...)

"""
    calc_convergence(envs::PullingThroughEnv)

Evaluate convergence measure for the pulling through algorithm as the elementwise difference
of the north and west lef- and right-gauging bond tensors.
"""
function calc_convergence(env::PullingThroughEnv)
    if isnothing(env.LN) || isnothing(env.RW)
        return one(scalartype(env.N))
    else
        return norm(env.LN - env.RW)
    end
end

#
# Symmetric leading boundary
#

"""
    MPSKit.leading_boundary([envinit], state, alg::PullingThrough)

Contract `state` using pulling through and return the environment. Per default, a random
initial environment is used.
"""
function MPSKit.leading_boundary(state, alg::PullingThrough)
    return MPSKit.leading_boundary(
        PullingThroughEnv(state, oneunit(spacetype(state))), state, alg
    )
end
function MPSKit.leading_boundary(envinit, state, alg::PullingThrough)
    # run the iterative algorithm
    env, N, ϵ = pulling_through_iterate(envinit, state, alg)

    # TODO: finalize environment and impose all the symmetries
    # TODO: implement fixed-point differentiation in symmetric gauge
    return env, N, ϵ
end
