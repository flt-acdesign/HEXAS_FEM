###########################################
# FILE: IterativeSolver.jl
###########################################

module IterativeSolver

using LinearAlgebra, Printf
using ..CPUSolver  # Use parent module's CPUSolver
using ..GPUSolver  # Use parent module's GPUSolver

export solve_system_iterative

"""
    solve_system_iterative(nodes, elements, E, nu, bc_indicator, f;
                          solver_type=:matrix_free, max_iter=1000, tol=1e-6,
                          use_precond=true, density=nothing,
                          gpu_method=:native, krylov_solver=:cg)

High-level interface that dispatches between different solver types:
- `:matrix_free` => matrix-free CG on CPU
- `:gpu` => assembled CSR solve on GPU (with options for native or Krylov solvers)

# Arguments
- `solver_type`: Type of solver to use (:matrix_free or :gpu)
- `max_iter`: Maximum iterations for iterative solvers
- `tol`: Convergence tolerance
- `use_precond`: Whether to use preconditioning (when available)
- `density`: Element density array (required for GPU solver)
- `gpu_method`: When using GPU, specifies the implementation (:native or :krylov)
- `krylov_solver`: When using Krylov.jl, specifies which algorithm (:cg, :minres, or :bicgstab)
"""
function solve_system_iterative(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                                bc_indicator::Matrix{T}, f::Vector{T};
                                solver_type=:matrix_free, max_iter=1000, tol=1e-6,
                                use_precond=true, density::Vector{T}=nothing,
                                gpu_method=:native, krylov_solver=:cg) where T

    if solver_type == :matrix_free
        return CPUSolver.solve_system_cpu(
            nodes, elements, E, nu, bc_indicator, f;
            max_iter=max_iter, tol=tol, use_precond=use_precond, density=density
        )
    elseif solver_type == :gpu
        if density === nothing
            error("You must provide a density array for GPU solver.")
        end
        return GPUSolver.solve_system_gpu(
            nodes, elements, E, nu, bc_indicator, f, density;
            max_iter=max_iter, tol=tol, 
            method=gpu_method, solver=krylov_solver, use_precond=use_precond
        )
    else
        error("Unknown solver type: $solver_type. Use :matrix_free or :gpu.")
    end
end

end  # module IterativeSolver