# Fermi-Hubbard model at half filling

using Revise

# # use slightly hacked forks of KrylovKit and OptimKit
# import Pkg
# Pkg.add(; url="https://github.com/leburgel/KrylovKit.jl", rev="lb/relax_realeigsolve")
# Pkg.add(; url="https://github.com/leburgel/OptimKit.jl", rev="lb/hack_backtracking")

using TensorKit
using PEPSKit
using MPSKitModels: hubbard_model, e_plusmin, e_minplus, e_number, e_number_updown
using OptimKit
using KrylovKit

include("../spatial_toolbox.jl")
include("../u1_toolbox.jl")
include("../space_shifting.jl") # TODO: remove this once it's merged into PEPSKit.jl; see https://github.com/QuantumKitHub/PEPSKit.jl/pull/135

# Setup
# -----

# specify symmetries
fermion = fℤ₂
particle_symmetry = U1Irrep
spin_symmetry = Trivial
S = fermion ⊠ particle_symmetry # symmetry sector

# define lattice and virtual spaces
lattice = InfiniteSquare(2, 2)
D = 1
Vpeps = Vect[S]((0, 0) => 2 * D, (1, 1) => D, (1, -1) => D)
χ = 2
Venv = Vect[S](
    (0, 0) => 4 * χ, (1, -1) => 2 * χ, (1, 1) => 2 * χ, (0, 2) => χ, (0, -2) => χ
)

# auxiliary physical space: shift the total U1 charge to impose a finite particle density
Saux = S((1, -1))

# define hamiltonian parameters; should give E = 4 * -0.5244140625...
U_test = 8
t_test = 1

# define algorithms
trscheme = FixedSpaceTruncation()
# trscheme = truncbelow(1e-4) & truncdim(5 * χ)
ctm_alg = SimultaneousCTMRG(; tol=1e-8, maxiter=500, verbosity=2, trscheme)
# gradient_alg = LinSolver(;
#     solver=GMRES(; tol=1e-6, maxiter=3, verbosity=3), iterscheme=:diffgauge
# )
gradient_alg = EigSolver(;
    solver=Arnoldi(; tol=1e-6, maxiter=30, verbosity=3, krylovdim=30, eager=true),
    iterscheme=:diffgauge,
)

reuse_env = true
# ls_alg = HagerZhangLineSearch(;
#     maxiter=4, maxfg=10, verbosity=5, c₁=0.01, c₂=0.99, ρ=2.0, ϵ=1e-3
# )
ls_alg = BackTrackingLineSearch(; c₁=1e-4, maxiter=10, maxfg=10, maxstep=5.0)
optimization_alg = LBFGS(
    10; acceptfirst=true, maxiter=500, gradtol=1e-4, verbosity=3, linesearch=ls_alg
)

## Initialize and shift Hamiltonian

H_t = hubbard_model(
    ComplexF64, particle_symmetry, spin_symmetry, lattice; t=t_test, U=U_test
)

H_t = add_physical_charge(H_t, fill(Saux, size(H_t.lattice)...))
Pspaces = H_t.lattice

# Part 0: use default PEPSKit.fixedpoint
# --------------------------------------

mode = "naive optimization with 2x2 unit cell"

@info "Running $mode"

psi0 = InfinitePEPS(randn, ComplexF64, Pspaces, fill(Vpeps, size(Pspaces)...))
env0 = CTMRGEnv(psi0, Venv)
env0, = leading_boundary(env0, psi0, ctm_alg)

pepsopt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=optimization_alg,
    gradient_alg=gradient_alg,
    reuse_env=reuse_env,
)
peps_final, env_final, cost, info = fixedpoint(H_t, psi0, env0, pepsopt_alg)

@info "Finished $mode"

numfg = info.fg_evaluations
E = result.cost
numiter = length(info.costs)

@info "Energy: $E\t numfg: $numfg\t numiter: $numiter"
