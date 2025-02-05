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
# TODO: do I need to write an explicit rrule for this?

function costfun(peps::InfinitePEPS, env::SymmetricEnv, O::LocalOperator)
    return costfun(peps, CTMRGEnv(env), O)
end

#
# Fixed-point equations
#

# fixed-point transfer functions
function fp_transfer_west(
    ::Val{:real},
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
) where {S}
    AU = physical_flip(A)
    @autoopt @tensor A´[χ_SE D_E_above D_E_below; χ_NE] :=
        A[χ_NNW D_N_above D_N_below; χ_NE] *
        AU[χ_WSW D_W_above D_W_below; χ_WNW] *
        AU[χ_SE D_S_above D_S_below; χ_SSW] *
        X[χ_WNW; χ_NNW] *
        X[χ_SSW; χ_WSW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
    return physical_flip(A´) # restore original space...
end

function fp_transfer_west(
    ::Val{:complex},
    A::MPSKit.GenericMPSTensor{S,3},
    X::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
) where {S}
    AU = physical_flip(A)
    @autoopt @tensor A´[χ_SE D_E_above D_E_below; χ_NE] :=
        A[χ_NNW D_N_above D_N_below; χ_NE] *
        AU[χ_WSW D_W_above D_W_below; χ_WNW] *
        conj(A[χ_SSW D_S_above D_S_below; χ_SE]) *
        X[χ_WNW; χ_NNW] *
        X[χ_SSW; χ_WSW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
    return physical_flip(A´) # restore original space...
end

# TODO: just remove distinction between real and complex

# 'rectangular' version of fixed-point equations
function pt_fixedpoint(::Val{:rectangular}, state, A, X, N)
    O = _local_sandwich(state)

    # Hermiticity
    FP1 = A - physical_flip(_conj(A))

    # Eigenvalue equation
    FP2 = fp_transfer_west(Val(:real), A, X, O) - N * absorb_bond_matrix(A, X)

    # Left fixed point condition
    FP3 = MPSKit.transfer_left(X^2, A, A) - X^2

    # Normalization
    FP4 = abs(tr(X^4)) - one(scalartype(X))

    return (FP1, FP2, FP3, FP4)
end

# 'square' version of fixed-point equations
function pt_fixedpoint(::Val{:square}, state, A, X, N)
    O = _local_sandwich(state)

    # Eigenvalue equation
    FP2´ = fp_transfer_west(Val(:complex), A, X, O) - N * absorb_bond_matrix(A, X)

    # Left fixed point condition
    FP3 = MPSKit.transfer_left(X^2, A, A) - X^2

    # Normalization
    FP4 = abs(tr(X^4)) - one(scalartype(X))

    return (FP2´, FP3, FP4)
end

# need some initial points...
_randomize!(A::AbstractTensorMap) = MPSKit.randomize!(A)
_randomize!(::T) where {T<:Number} = randn(T)
function initialize_rhs(F, state, A, X, N)
    return _randomize!.(pt_fixedpoint(F, state, A, X, N))
end

# TODO: figure out al the appropriate conditions and corresponding projections...

function project_hermitian(A::MPSKit.GenericMPSTensor)
    A´ = (A + physical_flip(_conj(A))) / 2
    return A´
end

#
# Derivatives
#

function _rrule(
    gradmode::LinSolver{:square},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state,
    alg::PullingThrough,
)
    env, N, ϵ = leading_boundary(envinit, state, alg)

    # attempt at version based on a square linear problem
    function leading_boundary_square_pullback(ΔX)
        ∂self = NoTangent()
        ∂env₀ = ZeroTangent()
        ∂alg = NoTangent()

        Δenv = unthunk(ΔX[1])
        ΔN = unthunk(ΔX[2])

        # use a tuple instead of the fancy struct, and unpack and repack at the end
        ΔA = Δenv.A
        ΔX = Δenv.X

        # DEBUGGING
        fps = pt_fixedpoint(Val(:square), state, env.A, env.X, N)
        nrm = sum(norm.(fps))
        nrm < alg.tol ||
            @warn "Fixed-point equations not satisfied, still using the gradient: $nrm"

        # find partial gradients of pulling through fixed-point equation
        f(state, A, X, N) = pt_fixedpoint(Val(:square), state, A, X, N)
        _, pt_vjp = pullback(f, state, env.A, env.X, N)

        function vjp_env(x)
            # apply pullback and unpack
            y = pt_vjp(x)
            ΔA = y[2]
            ΔX = y[3]
            ΔN = y[4]

            # TODO: impose all appropriate conditions
            # ΔA = project_hermitian(ΔA)
            # ...

            return (ΔA, ΔX, ΔN)
        end
        vjp_state(x) = pt_vjp(x)[1]

        # solve linear problem to invert environment pullback
        Δx = (ΔA, ΔX, N)
        x₀ = initialize_rhs(Val(:square), state, env.A, env.X, N)
        x, info = linsolve(vjp_env, Δx, x₀, gradmode.solver)
        if gradmode.solver.verbosity > 0 && info.converged != 1
            @warn(
                "gradient fixed-point iteration reached maximal number of iterations:", info
            )
        end

        # map solution to proper adjoint using state pullback
        ∂state = (-1) * vjp_state(x)

        return ∂self, ∂env₀, ∂state, ∂alg
    end

    return (env, N, ϵ), leading_boundary_square_pullback
end

function _rrule(
    gradmode::LSSolver{:rectangular},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state,
    alg::PullingThrough,
)
    env, N, ϵ = leading_boundary(envinit, state, alg)

    # attempt at version based on a rectangular linear problem
    function leading_boundary_rectangular_pullback(ΔX)
        ∂self = NoTangent()
        ∂env₀ = ZeroTangent()
        ∂alg = NoTangent()

        Δenv = unthunk(ΔX[1])
        ΔN = unthunk(ΔX[2])

        # use a tuple instead of the fancy struct, and unpack and repack at the end
        ΔA = Δenv.A
        ΔX = Δenv.X

        # DEBUGGING
        fps = pt_fixedpoint(Val(:rectangular), state, env.A, env.X, N)
        nrm = sum(norm.(fps))
        nrm < alg.tol ||
            @warn "Fixed-point equations not satisfied, still using the gradient: $nrm"

        # find partial gradients of pulling through fixed-point equation
        f(state, A, X, N) = pt_fixedpoint(Val(:rectangular), state, A, X, N)
        _, pt_vjp = pullback(f, state, env.A, env.X, N)

        function vjp_env(x)
            # apply pullback and unpack
            y = pt_vjp(x)
            ΔA = y[2]
            ΔX = y[3]
            ΔN = y[4]

            # TODO: impose all appropriate conditions
            # ΔA = project_hermitian(ΔA)
            # ...

            return (ΔA, ΔX, ΔN)
        end
        vjp_state(x) = pt_vjp(x)[1]

        # solve linear problem to invert environment pullback
        Δx = (ΔA, ΔX, N)
        # hack to get the adjoint action of the environment pullback
        _, vjp_env_adjoint = pullback(
            vjp_env, pt_fixedpoint(Val(:rectangular), state, env.A, env.X, N)
        )
        y, info = lssolve((vjp_env, vjp_env_adjoint), Δx, gradmode.solver)
        if gradmode.solver.verbosity > 0 && info.converged != 1
            @warn(
                "gradient fixed-point iteration reached maximal number of iterations:", info
            )
        end

        ∂state = (-1) * vjp_state(y)

        return ∂self, ∂env₀, ∂state, ∂alg
    end

    return (env, N, ϵ), leading_boundary_rectangular_pullback
end
