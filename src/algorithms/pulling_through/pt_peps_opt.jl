#
# Random thingies
#

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

# fixed-point transfer functions

# duplicate arguments for use in forward computation
function fp_transfer_west(
    ::Val{F}, A::MPSKit.GenericMPSTensor{S,3}, X::MPSKit.MPSBondTensor{S}, O::PEPSSandwich
) where {F,S}
    return fp_transfer_west(Val(F), A, A, A, X, X, O)
end

# expanded calls for use backward pass
function fp_transfer_west(
    ::Val{:real},
    AN::MPSKit.GenericMPSTensor{S,3},
    AW::MPSKit.GenericMPSTensor{S,3},
    AS::MPSKit.GenericMPSTensor{S,3},
    XNW::MPSKit.MPSBondTensor{S},
    XSW::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
) where {S}
    AWU = physical_flip(AW)
    ASU = physical_flip(AS)
    @autoopt @tensor AW´[χ_SE D_E_above D_E_below; χ_NE] :=
        AN[χ_NNW D_N_above D_N_below; χ_NE] *
        AWU[χ_WSW D_W_above D_W_below; χ_WNW] *
        ASU[χ_SE D_S_above D_S_below; χ_SSW] *
        XNW[χ_WNW; χ_NNW] *
        XSW[χ_SSW; χ_WSW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
    return physical_flip(AW´) # restore original space...
end

function fp_transfer_west(
    ::Val{:complex},
    AN::MPSKit.GenericMPSTensor{S,3},
    AW::MPSKit.GenericMPSTensor{S,3},
    AS::MPSKit.GenericMPSTensor{S,3},
    XNW::MPSKit.MPSBondTensor{S},
    XSW::MPSKit.MPSBondTensor{S},
    O::PEPSSandwich,
) where {S}
    AWU = physical_flip(AW)
    @autoopt @tensor AW´[χ_SE D_E_above D_E_below; χ_NE] :=
        AN[χ_NNW D_N_above D_N_below; χ_NE] *
        AWU[χ_WSW D_W_above D_W_below; χ_WNW] *
        conj(AS[χ_SSW D_S_above D_S_below; χ_SE]) *
        XNW[χ_WNW; χ_NNW] *
        XSW[χ_SSW; χ_WSW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
    return physical_flip(AW´) # restore original space...
end

# 'rectangular' version of fixed-point equations
function pt_fixedpoint(::Val{:rectangular}, network, A, X, N)
    O = network[1, 1]
    X2 = X * X
    X4 = X2 * X2

    # Hermiticity
    FP1 = A - physical_flip(_conj(A))

    # Eigenvalue equation
    FP2 = fp_transfer_west(Val(:real), A, X, O) - N * absorb_bond_matrices(A, X, X)

    # Left fixed point condition
    FP3 = MPSKit.transfer_left(X2, A, A) - X2

    # Normalization
    FP4 = abs(tr(X4)) - one(scalartype(X))

    return (FP1, FP2, FP3, FP4)
end

# 'square' version of fixed-point equations
function pt_fixedpoint(::Val{:square}, network, A, X, N)
    O = network[1, 1]
    X2 = X * X
    X4 = X2 * X2

    # Eigenvalue equation
    FP2´ = fp_transfer_west(Val(:complex), A, X, O) - N * absorb_bond_matrices(A, X, X)

    # Left fixed point condition
    FP3 = MPSKit.transfer_left(X2, A, A) - X2

    # Normalization
    FP4 = abs(tr(X4)) - one(scalartype(X))

    return (FP2´, FP3, FP4)
end

# need some initial points...
_randomize!(A::AbstractTensorMap) = MPSKit.randomize!(A)
_randomize!(::T) where {T<:Number} = randn(T)
function initialize_rhs(F, state, A, X, N)
    return _randomize!.(pt_fixedpoint(F, InfiniteSquareNetwork(state), A, X, N))
end

# TODO: figure out al the appropriate conditions and corresponding projections...

function project_hermitian(A::MPSKit.GenericMPSTensor)
    A´ = (A + physical_flip(_conj(A))) / 2
    return A´
end

#
# Derivatives
#

# partial pushforward implementing environment JVP
function generate_partial_pushforward(::Val{:rectangular}, network, A, X, N)
    O = network[1, 1]
    X2 = X * X

    function partial_pushforward((∂A, ∂X, ∂N))
        # Hermiticity
        ∂FP1 = ∂A - physical_flip(_conj(∂A))

        # Eigenvalue equation
        ∂FP2 =
            fp_transfer_west(Val(:real), ∂A, A, A, X, X, O) +
            fp_transfer_west(Val(:real), A, ∂A, A, X, X, O) +
            fp_transfer_west(Val(:real), A, A, ∂A, X, X, O) +
            fp_transfer_west(Val(:real), A, A, A, ∂X, X, O) +
            fp_transfer_west(Val(:real), A, A, A, X, ∂X, O) -
            N * absorb_bond_matrices(∂A, X, X) - N * absorb_bond_matrices(A, ∂X, X) -
            N * absorb_bond_matrices(A, X, ∂X)

        # Left fixed point condition
        ∂FP3 =
            MPSKit.transfer_left(∂X * X, A, A) +
            MPSKit.transfer_left(X * ∂X, A, A) +
            MPSKit.transfer_left(X2, ∂A, A) +
            MPSKit.transfer_left(X2, A, ∂A) - X * ∂X - ∂X * X

        # Normalization
        ∂FP4 = 4 * tr(X2 * X * ∂X)

        return (∂FP1, ∂FP2, ∂FP3, ∂FP4)
    end

    return partial_pushforward
end

# # partial pullback implementing environment VJP; TODO
# function generate_partial_pullback(::Val{:rectangular}, network, A, X, N)
#     O = network[1, 1]
#     X2 = X * X

#     function partial_pullback((ΔFP1, ΔFP2, ΔFP3, ΔFP4))
#         ΔA =

#         ΔX = # TODO

#         ΔN = # TODO

#         return (ΔA, ΔX, ΔN)
#     end

#     return partial_pullback
# end

# TODO: totally broken, need to figure out how to do this properly...
function _rrule(
    gradmode::LinSolver{:square},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state::InfinitePEPS,
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
        fps = pt_fixedpoint(Val(:square), InfiniteSquareNetwork(state), env.A, env.X, N)
        nrm = sum(norm.(fps))
        nrm < alg.tol ||
            @warn "Fixed-point equations not satisfied, still using the gradient: $nrm"

        # find partial gradients of pulling through fixed-point equation
        function f(state, A, X, N)
            return pt_fixedpoint(Val(:square), InfiniteSquareNetwork(state), A, X, N)
        end
        _, pt_vjp = pullback(f, state, env.A, env.X, N)

        function vjp_env(x)

            # TODO: impose all appropriate conditions
            # TODO: 'precondition' in some appropriate way?

            # apply pullback and unpack
            y = pt_vjp(x)
            ΔA = y[2]
            ΔX = y[3]
            ΔN = y[4]

            # TODO: impose all appropriate conditions
            # ΔA = project_hermitian(ΔA)?
            # ΔX is diagonal and real?
            # ...

            return (ΔA, ΔX, ΔN)
        end
        vjp_state(x) = pt_vjp(x)[1]

        # solve linear problem to invert environment pullback
        Δx = (ΔA, ΔX, N)
        x₀ = initialize_rhs(Val(:square), state, env.A, env.X, N)
        x, info = reallinsolve(vjp_env, Δx, x₀, gradmode.solver_alg)
        if gradmode.solver_alg.verbosity > 0 && info.converged != 1
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
    state::InfinitePEPS, # TODO: generalize this
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

        # initialize proper form of the fixed-point equations
        function f(state, A, X, N)
            return pt_fixedpoint(Val(:rectangular), InfiniteSquareNetwork(state), A, X, N)
        end
        # and evaluate everything in the primal fixed-point solution
        X0 = (state, env.A, env.X, N)

        # check if fixed-point equations are actually satisfied
        Y0 = f(X0...)
        nrm = sum(norm.(Y0))
        nrm < alg.tol ||
            @warn "Fixed-point equations not satisfied, still using the gradient: $nrm"

        # get the partial gradients of the fixed-point equations
        # start from the full automatic pullback
        _, pt_vjp = pullback(f, state, env.A, env.X, N)
        # restrict to environment pullback
        function vjp_env(x) # TODO: test against manual partial pullback...

            # TODO: impose all appropriate conditions
            # TODO: 'precondition' in some appropriate way?

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
        # hack to get the adjoint action of the environment pullback
        jvp_env = generate_partial_pushforward(
            Val(:rectangular), InfiniteSquareNetwork(state), env.A, env.X, N
        )
        # restrict to state pullback
        vjp_state(x) = pt_vjp(x)[1]

        # solve linear problem to invert environment pullback
        Δx = (ΔA, ΔX, N)
        y, info = reallssolve((vjp_env, jvp_env), Δx, gradmode.solver_alg)
        if gradmode.solver_alg.verbosity > 0 && info.converged != 1
            @warn(
                "gradient fixed-point iteration reached maximal number of iterations:", info
            )
        end

        ∂state = (-1) * vjp_state(y)

        return ∂self, ∂env₀, ∂state, ∂alg
    end

    return (env, N, ϵ), leading_boundary_rectangular_pullback
end
