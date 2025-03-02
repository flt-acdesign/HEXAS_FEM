###########################################
# FILE: CPUSolver.jl
###########################################

module CPUSolver

using LinearAlgebra, SparseArrays, Base.Threads, Printf
using ..Element

export MatrixFreeSystem, solve_system_cpu

# -----------------------------------------------------------------------
# Matrix-Free System Definition
# -----------------------------------------------------------------------
struct MatrixFreeSystem{T}
    nodes::Matrix{T}
    elements::Matrix{Int}
    E::T
    nu::T
    bc_indicator::Matrix{T}
    free_dofs::Vector{Int}
    constrained_dofs::Vector{Int}
    density::Vector{T}
end

"""
    MatrixFreeSystem(nodes, elements, E, nu, bc_indicator; density=nothing)

Build a matrix-free FE system. If density is not supplied, default to 1.0 per element.
"""
function MatrixFreeSystem(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, density::Vector{T}=nothing) where T
    nElem = size(elements, 1)
    if density === nothing
        density = ones(T, nElem)
    end

    nNodes = size(nodes, 1)
    ndof   = nNodes * 3
    constrained = falses(ndof)
    for i in 1:nNodes
        for j in 1:3
            if bc_indicator[i,j] > 0
                constrained[3*(i-1)+j] = true
            end
        end
    end

    free_dofs        = findall(!, constrained)
    constrained_dofs = findall(x->x, constrained)

    return MatrixFreeSystem(nodes, elements, E, nu, bc_indicator,
                            free_dofs, constrained_dofs, density)
end

# -----------------------------------------------------------------------
# Matrix-Vector Product (Matrix-Free CPU)
# -----------------------------------------------------------------------
function apply_stiffness(system::MatrixFreeSystem{T}, x::Vector{T}) where T
    nNodes = size(system.nodes, 1)
    ndof   = nNodes * 3
    nElem  = size(system.elements, 1)

    result = zeros(T, ndof)
    result_local = [zeros(T, ndof) for _ in 1:nthreads()]

    @threads for e in 1:nElem
        tid  = threadid()
        conn = system.elements[e,:]
        elem_nodes = system.nodes[conn,:]
        E_local = system.E * system.density[e]

        ke = Element.hex_element_stiffness(elem_nodes, E_local, system.nu)

        # Gather local x
        u_elem = zeros(T, 24)
        for i in 1:8
            node_id = conn[i]
            for d in 1:3
                u_elem[3*(i-1)+d] = x[3*(node_id-1)+d]
            end
        end

        # Compute local f = ke * u_elem
        f_elem = ke * u_elem

        # Scatter into global
        for i in 1:8
            node_id = conn[i]
            for d in 1:3
                result_local[tid][3*(node_id-1)+d] += f_elem[3*(i-1)+d]
            end
        end
    end

    # Sum partial results from each thread
    for r in result_local
        result .+= r
    end
    return result
end

function apply_system(system::MatrixFreeSystem{T}, x::Vector{T}) where T
    return apply_stiffness(system, x)
end

function apply_system_free_dofs(system::MatrixFreeSystem{T}, x_free::Vector{T}) where T
    nNodes = size(system.nodes, 1)
    ndof   = nNodes * 3
    x_full = zeros(T, ndof)
    x_full[system.free_dofs] = x_free
    result_full = apply_system(system, x_full)
    return result_full[system.free_dofs]
end

# -----------------------------------------------------------------------
# Preconditioner (Diagonal)
# -----------------------------------------------------------------------
function compute_diagonal_preconditioner(system::MatrixFreeSystem{T}) where T
    nNodes = size(system.nodes, 1)
    ndof   = nNodes*3
    nElem  = size(system.elements, 1)
    diag   = zeros(T, ndof)
    diag_local = [zeros(T, ndof) for _ in 1:nthreads()]

    @threads for e in 1:nElem
        tid  = threadid()
        conn = system.elements[e,:]
        elem_nodes = system.nodes[conn,:]
        E_local = system.E * system.density[e]

        ke = Element.hex_element_stiffness(elem_nodes, E_local, system.nu)
        for i in 1:8
            node_id = conn[i]
            for d in 1:3
                diag_local[tid][3*(node_id-1)+d] += ke[3*(i-1)+d, 3*(i-1)+d]
            end
        end
    end

    for d in diag_local
        diag .+= d
    end
    return diag
end

# -----------------------------------------------------------------------
# Matrix-Free Conjugate Gradient (CPU)
# -----------------------------------------------------------------------
function matrix_free_cg_solve(system::MatrixFreeSystem{T}, f::Vector{T};
                              max_iter=1000, tol=1e-6, use_precond=true) where T
    f_free = f[system.free_dofs]
    n_free = length(system.free_dofs)
    x_free = zeros(T, n_free)

    diag_full = compute_diagonal_preconditioner(system)
    diag_free = diag_full[system.free_dofs]

    r = copy(f_free)
    z = use_precond ? r ./ diag_free : copy(r)
    p = copy(z)
    rz_old = dot(r, z)

    println("Starting matrix-free CG solve with $(n_free) unknowns...")
    total_time = 0.0
    for iter in 1:max_iter
        iter_start = time()
        Ap = apply_system_free_dofs(system, p)
        alpha = rz_old / dot(p, Ap)
        x_free .+= alpha .* p
        r .-= alpha .* Ap
        res_norm = norm(r) / norm(f_free)
        iter_time = time() - iter_start
        total_time += iter_time

        @printf("Iteration %d, residual = %.6e, time = %.6f sec, total time = %.6f sec\n",
                iter, res_norm, iter_time, total_time)

        if res_norm < tol
            println("CG converged in $iter iterations, residual = $res_norm, total time = $total_time sec")
            break
        end
        z = use_precond ? r ./ diag_free : copy(r)
        rz_new = dot(r, z)
        beta = rz_new / rz_old
        p .= z .+ beta .* p
        rz_old = rz_new
    end

    x_full = zeros(T, length(f))
    x_full[system.free_dofs] = x_free
    return x_full
end

"""
    solve_system_cpu(nodes, elements, E, nu, bc_indicator, f;
                     max_iter=1000, tol=1e-6, use_precond=true,
                     density=nothing)

High-level interface for the matrix-free CG solver on CPU.
"""
function solve_system_cpu(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, f::Vector{T};
                          max_iter=1000, tol=1e-6, use_precond=true,
                          density::Vector{T}=nothing) where T
    
    system = MatrixFreeSystem(nodes, elements, E, nu, bc_indicator, density)
    solve_start = time()
    solution = matrix_free_cg_solve(system, f, max_iter=max_iter, tol=tol, use_precond=use_precond)
    solve_end = time()
    @printf("Total solution time (matrix-free CPU): %.6f sec\n", solve_end - solve_start)
    return solution
end

end  # module CPUSolver