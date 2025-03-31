#
# Random thingies
#

"""
    struct PTLSSolver(; solver=KrylovKit.GMRES(), iterscheme=:rectangular) <: GradMode{iterscheme}

Gradient mode wrapper around `KrylovKit.LeastSquaresSolver` for solving the gradient linear
problem using iterative solvers.

Use for computing the pulling-through fixed-point gradient, where the `iterscheme` encodes
the formulation of the fixed-point equations that is used (:rectangular or :square).
"""
struct PTLSSolver{G,S}
    solver_alg::KrylovKit.LeastSquaresSolver
end
function PTLSSolver(;
    solver_alg=KrylovKit.LSMR(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol),
    gauge=:center,
    style=:naive,
)
    return PTLSSolver{gauge,style}(solver_alg)
end

# TODO: get rid of this hack
_alg_or_nt(::Type{CTMRGAlgorithm}, alg::A) where {A<:PullingThrough} = alg

#
# Expectation values
#

# aka lazy hacks 
function CTMRGEnv(env::SymmetricEnv{C,T}) where {C,T}
    TN = env.A
    TE = env.A
    TS = physical_flip(env.A)
    TW = physical_flip(env.A)

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
# TODO: do I need to write an explicit rrule for this? seem like not...

function cost_function(peps::InfinitePEPS, env::SymmetricEnv, O::LocalOperator)
    return cost_function(peps, CTMRGEnv(env), O)
end

#
# Fixed-point equations
#

function pt_fixedpoint(::Val{Gauge}, ::Val{Style}, network, A, X, N) where {Gauge,Style}
    O = network[1, 1]

    # Hermiticity
    FP1 = fixed_point_1(Val(Gauge), Val(Style), A, X)

    # Eigenvalue equation
    FP2 = fixed_point_2(Val(Gauge), Val(Style), A, X, N, O)

    # Left fixed point condition
    FP3 = fixed_point_3(Val(Gauge), Val(Style), A, X)

    # Normalization
    FP4 = fixed_point_4(Val(Gauge), Val(Style), X)

    return (FP1, FP2, FP3, FP4)
    # return (FP2, FP3, FP4)
end

#
# Derivatives
#

function _rrule(
    gradmode::PTLSSolver{Gauge,Style},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state::InfinitePEPS, # TODO: generalize this
    alg::PullingThrough,
) where {Gauge,Style}
    env, N, ϵ = leading_boundary(envinit, state, alg)

    # attempt at version based on a rectangular linear problem
    function leading_boundary_rectangular_pullback(Δx_)
        Δself = NoTangent()
        Δenv₀ = ZeroTangent()
        Δalg = NoTangent()

        Δenv, ΔN, _ = unthunk.(Δx_)
        if ΔN isa AbstractZero
            ΔN = zerovector(N) # TODO: better handling of ZeroTangents?
        end

        # unpack convenience struct and just use a tuples
        ΔA, ΔX = Δenv.A, Δenv.X

        # initialize proper form of the fixed-point equations
        function FP(state, A, X, N)
            return pt_fixedpoint(
                Val(Gauge), Val(Style), InfiniteSquareNetwork(state), A, X, N
            )
        end
        # and evaluate everything in the primal fixed-point solution
        x0 = (state, env.A, env.X, N)

        # check if fixed-point equations are actually satisfied
        fp_nrms = norm.(FP(x0...))
        sum(fp_nrms) < alg.tol ||
            @warn "Fixed-point equations not satisfied, still using the gradient: $fp_nrms"

        # get the partial gradients of the fixed-point equations

        # start from the full automatic pullback
        _, pt_vjp = pullback(FP, state, env.A, env.X, N)

        # DEBUGGING
        vjp_env2 = generate_partial_pullback(
            Val(Gauge), Val(Style), InfiniteSquareNetwork(state), env.A, env.X, N
        )

        # restrict to the pure environment pullback
        function vjp_env(ΔFP) # TODO: test against manual partial pullback...
            # apply pullback, isolate the environment part
            ΔA_, ΔX_, ΔN_ = pt_vjp(ΔFP)[2:end]
            # if Gauge == :center && Style == :naive # DEBUGGING
            #     ΔA__, ΔX__, ΔN__ = vjp_env2(ΔFP)
            #     @info norm.((ΔA_, ΔX_, ΔN_) .- (ΔA__, ΔX__, ΔN__))
            #     @info dot(ΔX_, ΔX__) / norm(ΔX_) / norm(ΔX__) # DEBUGGING
            #     # ΔA_, ΔX_, ΔN_ = ΔA__, ΔX__, ΔN__ # try the manual one instead?
            # end
            # Gauge == :center && (ΔA_ = project_hermitian(ΔA_)) # TODO: figure out project_hermitian
            return (ΔA_, ΔX_, ΔN_)
        end

        # manually implement the environment pushforward as the adjoint action of the
        # environment pullback
        jvp_env = generate_partial_pushforward(
            Val(Gauge), Val(Style), InfiniteSquareNetwork(state), env.A, env.X, N
        )

        # restrict to state pullback
        vjp_state(x) = pt_vjp(x)[1]

        # solve linear problem to invert environment pullback
        Δx = (ΔA, ΔX, ΔN)
        Δy, info = reallssolve((vjp_env, jvp_env), Δx, gradmode.solver_alg)
        if gradmode.solver_alg.verbosity > 0 && info.converged != 1
            @warn(
                "gradient fixed-point iteration reached maximal number of iterations:", info
            )
        end
        # check if the linear problem actually converged
        linres = vjp_env(Δy) .- Δx
        if norm(linres) > 1e2 * alg.tol
            msg = "gradient fixed-point iteration did not actually converge:"
            msg *= "\n  ‖ b - A x ‖ = $(norm(linres))"
            msg *= "\n  ‖ Aᴴ(b - A x) ‖ = $(norm(jvp_env(linres)))"
            msg *= "\n  norm.(b - A x) = $(norm.(linres))"
            @warn msg
        end

        # then plug it into the state pullback to get the state cotangent
        Δstate = (-1) * vjp_state(Δy)

        return Δself, Δenv₀, Δstate, Δalg
    end

    return (env, N, ϵ), leading_boundary_rectangular_pullback
end
