"""
    PullingThrough

Pulling-through contraction algorithm.
"""
@kwdef struct PullingThrough{F}
    tol::Float64 = Defaults.pt_tol
    maxiter::Int = Defaults.pt_maxiter
    miniter::Int = Defaults.pt_miniter
    verbosity::Int = Defaults.pt_verbosity
    finalize::F = Defaults._finalize

    dynamic_tols::Bool = Defaults.dynamic_tols
    alg_gauge = Defaults.pt_alg_gauge(; dynamic_tols)
    alg_eigsolve = MPSKit.Defaults.alg_eigsolve(;
        ishermitian=false, tol=1e-14, tol_factor=Defaults.eigs_tolfactor, dynamic_tols
    )
end

#
# Iterative contraction routine
#

"""
    pulling_through_update(state, env, alg::PullingThrough) -> env′, info

Perform a single pulling through iteration.
"""
function pulling_through_update(
    network::InfiniteSquareNetwork, env::PullingThroughEnv, alg_eigsolve, alg_gauge
)
    # update west
    env = gauge_north(env, alg_gauge)
    _, W_next = MPSKit.fixedpoint(env.W, :LM, alg_eigsolve) do x
        return transfer_west(x, env.NL, network[1, 1])
    end
    @reset env.W = W_next
    # @reset env.W = normalize_mps(W_next)

    # update north
    env = gauge_west(env, alg_gauge)
    λ, N_next = MPSKit.fixedpoint(env.N, :LM, alg_eigsolve) do x
        return transfer_north(x, env.WR, network[1, 1])
    end
    @reset env.N = N_next
    # @reset env.N = normalize_mps(N_next)

    return env, λ
end

"""
    pulling_through_iterate(env, state, alg::PullingThrough)

Converge a northwest pulling through corner for a given state.
"""
function pulling_through_iterate(
    envinit::PullingThroughEnv, network::InfiniteSquareNetwork, alg::PullingThrough
)
    ϵ::Float64 = calc_convergence(envinit)
    N = 0.0
    env = deepcopy(envinit)
    log = ignore_derivatives(() -> MPSKit.IterLog("PT"))

    LoggingExtras.withlevel(; alg.verbosity) do
        pt_loginit!(log, ϵ, N)
        for iter in 1:(alg.maxiter)
            alg_eigsolve = MPSKit.updatetol(alg.alg_eigsolve, iter, ϵ)
            alg_gauge = MPSKit.updatetol(alg.alg_gauge, iter, ϵ)

            env, N = pulling_through_update(network, env, alg_eigsolve, alg_gauge)

            ϵ = calc_convergence(env)

            env = alg.finalize(iter, env, network)

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

    # normalize one more time at the end
    @reset env.N = normalize_mps(env.N)
    @reset env.W = normalize_mps(env.W)

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
function MPSKit.leading_boundary(
    envinit::PullingThroughEnv, network::InfiniteSquareNetwork, alg::PullingThrough
)
    # run the iterative algorithm
    env, N, ϵ = pulling_through_iterate(envinit, network, alg)

    # gauge-fix and symmetrize
    env, = symmetric_environment(env)

    # TODO: temporarily unpack SymmetricEnv to avoid issues?
    return env, N, ϵ
end
function leading_boundary(env₀, state, alg::PullingThrough)
    return leading_boundary(env₀, InfiniteSquareNetwork(state), alg)
end
function MPSKit.leading_boundary(
    env₀::SymmetricEnv, state::InfiniteSquareNetwork, alg::PullingThrough
)
    return MPSKit.leading_boundary(PullingThroughEnv(env₀), state, alg)
end
