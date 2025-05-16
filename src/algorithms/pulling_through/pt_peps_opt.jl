#
# Random thingies
#

# pulling through gradient solvers, with some extra type parameters for now...
abstract type PTGradMode{G,S} end

struct PTLSSolver{G,S} <: PTGradMode{G,S}
    solver_alg::KrylovKit.LeastSquaresSolver
end
function PTLSSolver(;
    solver_alg=KrylovKit.LSMR(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol),
    gauge=:center,
    style=:naive,
)
    return PTLSSolver{gauge,style}(solver_alg)
end

# TODO: remove this, since it doesn't really make any sense...
struct PTLinSolver{G,S} <: PTGradMode{G,S}
    solver_alg::KrylovKit.LinearSolver
end
function PTLinSolver(;
    solver_alg=KrylovKit.GMRES(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol),
    gauge=:center,
    style=:naive,
)
    return PTLinSolver{gauge,style}(solver_alg)
end

# TODO: get rid of this hack
_alg_or_nt(::Type{CTMRGAlgorithm}, alg::A) where {A<:PullingThrough} = alg
_alg_or_nt(::Type{GradMode}, alg::A) where {A<:PTGradMode} = alg

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

    # # Hermiticity -> removed in favor of explicit projection
    # FP1 = fixed_point_1(Val(Gauge), Val(Style), A, X)

    # Eigenvalue equation
    FP2 = fixed_point_2(Val(Gauge), Val(Style), A, X, N, O)

    # Left fixed point condition
    FP3 = fixed_point_3(Val(Gauge), Val(Style), A, X)

    # Normalization
    FP4 = fixed_point_4(Val(Gauge), Val(Style), X)

    # return (FP1, FP2, FP3, FP4)
    return (FP2, FP3, FP4)
end

# inner product for cotangent vectors at a given fixed-point
function cotangent_inner(Xfp::MPSKit.MPSBondTensor)
    # TODO: figure out the exact preconditioning we want...
    Xprec = inv(Xfp)^2
    # Xprec = Xfp^2

    function dotf(
        (ΔA1, ΔX1, ΔN1)::Tuple{TA,TX,TN}, (ΔA2, ΔX2, ΔN2)::Tuple{TA,TX,TN}
    ) where {TA<:MPSKit.GenericMPSTensor,TX<:MPSKit.MPSBondTensor,TN<:Number}
        # stupid first trial...
        ΔA1ΔA2 = dot(ΔA1, PEPSKit.absorb_bond_matrices(ΔA2, Xprec, Xprec))
        ΔX1ΔX2 = tr(ΔX1' * ΔX2 * Xprec) / 4
        ΔN1ΔN2 = dot(ΔN1, ΔN2)

        return ΔA1ΔA2 + ΔX1ΔX2 + ΔN1ΔN2
    end
    return dotf
end

# define appropriate inner product for the cotangent linear problem input and output space
function get_inner_products(::Val{:naive}, ::MPSKit.MPSBondTensor)
    real_inner = real ∘ dot
    return real_inner, real_inner
end
function get_inner_products(::Val{:regularized}, X::MPSKit.MPSBondTensor)
    fp_dot = cotangent_inner(X)
    input_inner = real ∘ dot
    output_inner = real ∘ fp_dot
    return input_inner, output_inner
end

function project_symmetric!(state::InfinitePEPS)
    @assert length(state) == 1 "No pulling-through gradients for larger unit cells"
    state[1] = PEPSKit.herm_depth_inv(PEPSKit.rot_inv(state[1]))
    return state
end

#
# Derivatives
#

function _rrule(
    gradmode::PTGradMode{Gauge,Style},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state::InfinitePEPS, # TODO: generalize this
    alg::PullingThrough,
) where {Gauge,Style}
    env, N, ϵ = leading_boundary(envinit, state, alg)

    # unpack for convenience
    A = env.A
    X = env.X

    # get the appropriate inner products
    input_inner, output_inner = get_inner_products(Val(Style), X)

    function leading_boundary_pullback(Δx_)
        Δself = NoTangent()
        Δenv₀ = ZeroTangent()
        Δalg = NoTangent()

        Δenv, ΔN, _ = unthunk.(Δx_)
        if ΔN isa AbstractZero
            ΔN = zero(scalartype(A)) # TODO: better handling of ZeroTangents and mixed scalartypes?
        end

        # unpack convenience struct and just use tuples
        ΔA, ΔX = Δenv.A, Δenv.X

        if Gauge == :center
            # toggle Hermitian projections
            ΔA = project_hermitian(ΔA)
            ΔX = project_hermitian(ΔX)
            if X isa DiagonalTensorMap && scalartype(X) <: Real
                # toggle real diagonal projection of corner cotangent
                ΔX = real(DiagonalTensorMap(ΔX))
            end
        end

        # need to re-promote the scalartype to avoid issues with mixed tuple scalartypes...
        # TODO: a more elegant scalartype promotion?
        if scalartype(ΔX) != scalartype(ΔA)
            if ΔX isa DiagonalTensorMap
                ΔX = DiagonalTensorMap(convert(storagetype(ΔA), ΔX.data), only(domain(ΔX)))
            elseif ΔX isa TensorMap
                ΔX = TensorMap(convert(storagetype(ΔA), ΔX.data), space(ΔX))
            end
        end

        # initialize proper form of the fixed-point equations
        function FP(state, A, X, N)
            return pt_fixedpoint(
                Val(Gauge), Val(Style), InfiniteSquareNetwork(state), A, X, N
            )
        end

        # check if fixed-point equations are actually satisfied
        FPS = FP(state, A, X, N)
        fp_nrms = norm.(FPS)
        sum(fp_nrms) < alg.tol ||
            @warn "Fixed-point equations not satisfied, still using the gradient: $fp_nrms"

        # get the partial gradients of the fixed-point equations

        # start from the full automatic pullback
        _, pt_vjp = pullback(FP, state, A, X, N)

        # restrict to the pure environment pullback, with pre- and post-projections
        function vjp_env(
            (ΔFP2, ΔFP3, ΔFP4)::Tuple{TFP2,TFP3,TFP4}
        ) where {TFP2<:MPSKit.GenericMPSTensor,TFP3<:MPSKit.MPSBondTensor,TFP4<:Number}
            if Gauge == :center
                # toggle Hermitian projections
                ΔFP2 = project_hermitian(ΔFP2)
                ΔFP3 = project_hermitian(ΔFP3)
            end

            # apply pullback, isolate the environment part
            ΔA_, ΔX_, ΔN_ = pt_vjp((ΔFP2, ΔFP3, ΔFP4))[2:end]

            if Gauge == :center
                # toggle Hermitian projections
                ΔA_ = project_hermitian(ΔA_)
                ΔX_ = project_hermitian(ΔX_)
                if X isa DiagonalTensorMap && scalartype(X) <: Real
                    # toggle real diagonal projection of corner cotangent
                    ΔX_ = DiagonalTensorMap(ΔX_)
                end
            end

            return (ΔA_, ΔX_, ΔN_)
        end

        # but make sure it works with the appropriate inner products
        function vjp_env(ΔFPS::InnerProductVec) # TODO: fix closure issue, very annoying...
            return InnerProductVec(vjp_env(ΔFPS.vec), output_inner)
        end

        # manually implement the environment pushforward as the adjoint action of the
        # environment pullback
        jvp_env = generate_environment_pushforward(
            Val(Gauge), Val(Style), InfiniteSquareNetwork(state), A, X, N
        )
        # but make sure it works with the appropriate inner products
        function jvp_envvv(∂env::InnerProductVec) # TODO: fix closure issue...
            return InnerProductVec(jvp_env(∂env.vec), input_inner)
        end

        # restrict to state pullback
        vjp_state(x) = pt_vjp(x)[1]

        # solve linear problem to invert environment pullback
        Δx = InnerProductVec((ΔA, ΔX, ΔN), output_inner)
        Δy, info = lssolve((vjp_env, jvp_envvv), Δx, gradmode.solver_alg)
        if gradmode.solver_alg.verbosity > 0 && info.converged != 1
            @warn(
                "gradient fixed-point iteration reached maximal number of iterations:", info
            )
        end
        # check if the linear problem actually converged
        linres = add(Δx, vjp_env(Δy), -1)
        if norm(linres) > 1e2 * alg.tol
            msg = "gradient fixed-point iteration did not actually converge:"
            msg *= "\n  ‖ b - A x ‖ = $(norm(linres))"
            msg *= "\n  ‖ Aᴴ(b - A x) ‖ = $(norm(jvp_envvv(linres)))"
            msg *= "\n  Euclidean: norm.(b - A x) = $(norm.(linres[]))"
            msg *= "\n  Euclidean: ‖ x ‖ = $(norm.(Δy[]))"
            @warn msg
        end

        # then plug it into the state pullback to get the state cotangent
        Δstate = (-1) * vjp_state(Δy[])
        # check symmetries on the resuting state cotangent; TODO: should this be manifest?
        @info "State cotangent norm right after solving linear problem: norm(Δstate)=$(norm(Δstate))"
        Δstate = project_symmetric!(Δstate)
        @info "State cotangent norm after projecting onto symmetric component: norm(Δstate)=$(norm(Δstate))"

        return Δself, Δenv₀, Δstate, Δalg
    end

    return (env, N, ϵ), leading_boundary_pullback
end
