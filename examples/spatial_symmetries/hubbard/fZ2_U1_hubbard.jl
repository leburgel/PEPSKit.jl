# Fermi-Hubbard model at half filling

using Revise

using TensorKit
using PEPSKit
using MPSKit: add_physical_charge
using OptimKit
using KrylovKit

# Setup
# -----

# specify symmetries
fermion = fℤ₂
particle_symmetry = U1Irrep
spin_symmetry = Trivial
S = fermion ⊠ particle_symmetry # symmetry sector

# define lattice and virtual spaces
latt = InfiniteSquare(2, 2)
D = 1
Vpeps = Vect[S]((0, 0) => 2 * D, (1, 1) => D, (1, -1) => D)
χ = 2
Venv = Vect[S](
    (0, 0) => 4 * χ, (1, -1) => 2 * χ, (1, 1) => 2 * χ, (0, 2) => χ, (0, -2) => χ
)

# auxiliary physical space: shift the total U1 charge to impose a finite particle density
Saux = S((1, -1))

# define hamiltonian parameters; should give E = 4 * -0.5244140625...
U = 8.0
t = 1.0

# define algorithms
optim_maxiter = 100
trscheme = FixedSpaceTruncation()
boundary_alg = SimultaneousCTMRG(; tol=1e-8, maxiter=500, verbosity=2, trscheme)
gradient_alg = EigSolver(;
    solver_alg=Arnoldi(; tol=1e-6, maxiter=30, verbosity=2, krylovdim=30, eager=true),
    iterscheme=:diffgauge,
)
reuse_env = true
ls_alg = BackTrackingLineSearch(; c₁=1e-4, maxiter=10, maxfg=10, maxstep=5.0)
optimizer_alg = LBFGS(
    10;
    acceptfirst=true,
    maxiter=optim_maxiter,
    gradtol=1e-4,
    verbosity=3,
    linesearch=ls_alg,
)

## Initialize and shift Hamiltonian

H_t = hubbard_model(ComplexF64, particle_symmetry, spin_symmetry, latt; t, U)
H_t = add_physical_charge(H_t, fill(Saux, size(H_t.lattice)...))
Pspaces = H_t.lattice

# Part 0: use default PEPSKit.fixedpoint
# --------------------------------------

mode = "naive optimization with 2x2 unit cell"

@info "Running $mode"

psi0 = InfinitePEPS(randn, ComplexF64, Pspaces, fill(Vpeps, size(Pspaces)...))
env0 = CTMRGEnv(psi0, Venv)
env0, = leading_boundary(env0, psi0, boundary_alg)

pepsopt_alg = PEPSOptimize(; boundary_alg, optimizer_alg, gradient_alg, reuse_env)
peps_final, env_final, E, info = fixedpoint(
    H_t, psi0, env0; boundary_alg, gradient_alg, optimizer_alg
)

@info "Finished $mode"

numfg = info.fg_evaluations
numiter = length(info.costs)

@info "Energy: $E\t numfg: $numfg\t numiter: $numiter"
