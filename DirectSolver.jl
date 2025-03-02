###########################################
# FILE: DirectSolver.jl
###########################################

module DirectSolver

using LinearAlgebra, SparseArrays, Base.Threads, Printf
using ..Element
using ..Boundary
using ..Mesh

export solve_system

"""
    assemble_global_stiffness_parallel(nodes, elements, E, nu, density)

Assembles the global stiffness matrix in sparse format by looping over all elements.
Now each element's Young's modulus is scaled by `density[e]`.
"""
function assemble_global_stiffness_parallel(nodes::Matrix{Float32},
                                            elements::Matrix{Int},
                                            E::Float32,
                                            nu::Float32,
                                            density::Vector{Float32})
    nNodes = size(nodes, 1)
    ndof   = nNodes * 3
    nElem  = size(elements, 1)

    local_triplets = [Vector{Tuple{Int,Int,Float32}}() for _ in 1:nthreads()]

    @threads for e in 1:nElem
        tid  = threadid()
        conn = elements[e, :]

        # Coordinates of the 8 nodes for this element
        elem_nodes = nodes[conn, :]

        # Scale E by density
        E_local = E * density[e]

        # Compute element stiffness
        ke = Element.hex_element_stiffness(elem_nodes, E_local, nu)

        # Accumulate (I, J, V) for global K
        for i in 1:8
            I = 3*(conn[i]-1)+1
            for j in 1:8
                J = 3*(conn[j]-1)+1
                for a in 1:3, b in 1:3
                    push!(local_triplets[tid],
                          (I+a-1, J+b-1, ke[3*(i-1)+a, 3*(j-1)+b]))
                end
            end
        end
    end

    all_triplets = vcat(local_triplets...)
    I_vec = [t[1] for t in all_triplets]
    J_vec = [t[2] for t in all_triplets]
    V_vec = [t[3] for t in all_triplets]

    # Build sparse matrix
    K_global = sparse(I_vec, J_vec, V_vec, ndof, ndof)
    return K_global
end

"""
    solve_system(nodes, elements, E, nu, bc_indicator, f; density=nothing)

High-level interface for the direct solver:
1. Assembly of K
2. Boundary condition reduction
3. Solve
4. Reconstruct full displacement

If `density` is not provided, it defaults to 1.0 for all elements.
"""
function solve_system(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                      bc_indicator::Matrix{T}, f::Vector{T};
                      density::Vector{T}=nothing) where T

    nElem = size(elements,1)
    if density === nothing
        density = ones(T, nElem)
    end

    nNodes = size(nodes, 1)
    ndof   = nNodes * 3

    # Identify constrained DoFs
    constrained = falses(ndof)
    for i in 1:nNodes
        for j in 1:3
            if bc_indicator[i,j] > 0
                constrained[3*(i-1)+j] = true
            end
        end
    end
    free_dofs = findall(!, constrained)

    # Assemble the global stiffness matrix
    K_global = assemble_global_stiffness_parallel(nodes, elements, E, nu, density)

    # Reduce system
    K_reduced = K_global[free_dofs, free_dofs]
    F_reduced = f[free_dofs]

    # Solve with LU
    println("Solving linear system via LU factorization (CPU Direct).")
    U_reduced = K_reduced \ F_reduced

    # Reconstruct full solution vector
    U_full = zeros(T, ndof)
    U_full[free_dofs] = U_reduced

    return U_full
end

end  # module DirectSolver
