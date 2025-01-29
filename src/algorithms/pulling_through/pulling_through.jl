_get_tol(x) = x.tol
_get_tol(x::MPSKit.DynamicTol) = x.alg.tol

"""
    PullingThrough

Pulling-through contraction algorithm.
"""
@kwdef struct PullingThrough{F}
    tol::Float64 = Defaults.ctmrg_tol
    maxiter::Int = Defaults.ctmrg_maxiter
    verbosity::Int = 1
    finalize::F = Defaults._finalize

    dynamic_tols::Bool = true
    alg_gauge = MPSKit.Defaults.alg_gauge(; verbosity=1, maxiter=100, dynamic_tols)
    alg_eigsolve = MPSKit.Defaults.alg_eigsolve(; ishermitian=false, dynamic_tols)
end

#
# Iterative contraction routine
#

# TODO: generalize this to a `getindex` on a 'contractible' network and adopt it everywhere...
_local_sandwich(state::InfinitePEPS) = (only(state.A), only(state.A))
_local_sandwich(state::InfinitePartitionFunction) = only(state.A)

"""
    pulling_through_update(state, env, alg::PullingThrough) -> env′, info

Perform a single pulling through iteration.
"""
function pulling_through_update(
    state::InfiniteSquareNetwork, env::PullingThroughEnv, alg_eigsolve, alg_gauge
)
    # update west
    env = gauge_north(env, alg_gauge)
    _, W_next = MPSKit.fixedpoint(env.W, :LM, alg_eigsolve) do x
        return transfer_west(x, env.NL, _local_sandwich(state))
    end
    @reset env.W = W_next
    # @reset env.W = normalize_mps(W_next)

    # update north
    env = gauge_west(env, alg_gauge)
    λ, N_next = MPSKit.fixedpoint(env.N, :LM, alg_eigsolve) do x
        return transfer_north(x, env.WR, _local_sandwich(state))
    end
    @reset env.N = N_next
    # @reset env.N = normalize_mps(N_next)

    return env, λ
end

"""
    pulling_through_iterate(env, state, alg::PullingThrough)

Converge a northwest pulling through corner for a given state.
"""
function pulling_through_iterate(envinit, state, alg::PullingThrough)
    ϵ::Float64 = calc_convergence(envinit)
    N = 0.0
    env = deepcopy(envinit)
    log = ignore_derivatives(() -> MPSKit.IterLog("PT"))

    LoggingExtras.withlevel(; alg.verbosity) do
        pt_loginit!(log, ϵ, N)
        for iter in 1:(alg.maxiter)
            alg_eigsolve = MPSKit.updatetol(alg.alg_eigsolve, iter, ϵ)
            alg_gauge = MPSKit.updatetol(alg.alg_gauge, iter, ϵ)

            env, N = pulling_through_update(state, env, alg_eigsolve, alg_gauge)

            ϵ = calc_convergence(env)

            env = alg.finalize(iter, env, state)

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
function MPSKit.leading_boundary(state, alg::PullingThrough)
    return MPSKit.leading_boundary(
        PullingThroughEnv(state, oneunit(spacetype(state))), state, alg
    )
end
function MPSKit.leading_boundary(envinit::SymmetricEnv, state, alg::PullingThrough)
    return MPSKit.leading_boundary(PullingThroughEnv(envinit), state, alg)
end
function MPSKit.leading_boundary(envinit::PullingThroughEnv, state, alg::PullingThrough)
    # run the iterative algorithm
    env, N, ϵ = pulling_through_iterate(envinit, state, alg)

    # gauge-fix and symmetrize
    env, = symmetric_environment(env)

    return env, N, ϵ
end

#
# PEPS optimization
#

function symm_peps_retract(x, η, α)
    peps = deepcopy(x[1])
    peps.A .+= η.A .* α
    env = deepcopy(x[2])
    return (peps, env), η
end

# hacks hacks hacks...

# I'm not reinventing hot water...
function CTMRGEnv(env::SymmetricEnv{C,T}) where {C,T}
    TN = env.A
    TE = env.A
    TS = apply_physical_operator(env.A, env.U)
    TW = apply_physical_operator(env.A, env.U)

    Ts = [TN, TE, TS, TW]
    Cs = [env.X, env.X, env.X, env.X]

    edges = Zygote.Buffer(Array{T,3}(undef, 4, 1, 1))
    corners = Zygote.Buffer(Array{C,3}(undef, 4, 1, 1))
    for dir in 1:4
        edges[dir, 1, 1] = Ts[dir]
        corners[dir, 1, 1] = Cs[dir]
    end
    return CTMRGEnv(copy(corners), copy(edges))
end

# lazy lazy lazy...
function costfun(peps::InfinitePEPS, env::SymmetricEnv, O::LocalOperator)
    return costfun(peps, CTMRGEnv(env), O)
end

# rrulez, hopefully...
function _rrule(
    gradmode::GradMode{F},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state,
    alg::PullingThrough,
) where {F}
    env, N, ϵ = leading_boundary(envinit, state, alg)

    function leading_boundary_real_pullback(ΔX)
        ∂self = NoTangent()
        ∂env₀ = ZeroTangent()
        ∂alg = NoTangent()

        Δenv = unthunk(ΔX[1])
        ΔN = unthunk(ΔX[2]) # this one is always ZeroTangent, which makes sense
        # the only problem is I don't know how to handle it...

        # attempt 1: adjoint of norm is a ZeroTangent
        if ΔN isa AbstractZero
            # find partial gradients of pulling through fixed-point equation,
            # but hack to consider the eigenvalue a constant...
            f_ez(A, env) = pt_fixedpoint(Val(F), A, env, N, env.U)[1]

            # DEBUGGING
            fps = pt_fixedpoint(Val(F), state, env, N, env.U)
            nrm = sum(norm.(fps))
            nrm < alg.tol ||
                @warn "Fixed-point equations not satisfied, still using the gradient: $nrm"

            _, fp_pullbacks_ez = rrule_via_ad(config, f_ez, state, env)

            ∂f∂A_ez(x)::typeof(state) = fp_pullbacks_ez(x)[2]
            ∂f∂x_ez(x)::typeof(env) = fp_pullbacks_ez(x)[3]
            ∂state = pt_fpgrad(Δenv, ∂f∂x_ez, ∂f∂A_ez, Δenv, gradmode)

            return ∂self, ∂env₀, ∂state, ∂alg
        else
            @warn "We shouldn't be here for now..."
            # TODO: figure out how to handle the general case

            # find partial gradients of pulling through fixed-point equation
            f_hrd(A, (env, N)) = pt_fixedpoint(Val(F), A, env, N, env.U)
            _, fp_pullbacks_hrd = rrule_via_ad(config, f_hrd, state, (env, N))

            ∂f∂A_hrd(x)::typeof(state) = fp_pullbacks_hrd(x)[2]
            # ∂f∂x(x)::typeof((Δenv, ΔN)) = fp_pullbacks(x)[3]
            ∂f∂x_hrd(x) = fp_pullbacks_hrd(x)[3] # TODO: fix issues with the output type conversion
            ∂state = pt_fpgrad((Δenv, ΔN), ∂f∂x_hrd, ∂f∂A_hrd, (Δenv, ΔN), gradmode)

            return ∂self, ∂env₀, ∂state, ∂alg
        end
    end

    return (env, N, ϵ), leading_boundary_real_pullback
end

function pt_fixedpoint(::Val{:real}, A, x, N, U)
    O = _local_sandwich(A)

    # dA
    dA = fp_transfer_west(Val(:real), x.A, x.X, U, O) - N * absorb_bond_matrix(x.A, x.X)

    # dX
    dX = MPSKit.transfer_left(x.X^2, x.A, x.A) - x.X^2

    # dN
    dN = abs(tr(x.X^4)) - one(scalartype(x.X))

    # group and return
    return SymmetricEnv(dX, dA, U), dN
end
function pt_fixedpoint(::Val{:complex}, A, x, N, U)
    O = _local_sandwich(A)

    # dA
    dA = fp_transfer_west(Val(:complex), x.A, x.X, U, O) - N * absorb_bond_matrix(x.A, x.X)

    # dX
    dX = MPSKit.transfer_left(x.X' * x.X, x.A, x.A) - x.X' * x.X

    # dN
    dN = abs(tr(x.X^4)) - one(scalartype(x.X))

    # group and return
    return SymmetricEnv(dX, dA, U), dN
end

function pt_fpgrad(Δx, ∂f∂x, ∂f∂A, y₀, alg::LinSolver)
    y, info = linsolve(∂f∂x, Δx, y₀, alg.solver)
    if alg.solver.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return -∂f∂A(y)
end
