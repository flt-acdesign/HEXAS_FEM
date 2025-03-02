###########################################
# FILE: GPUSolver.jl
###########################################

module GPUSolver

using LinearAlgebra, SparseArrays, Printf
using CUDA                   # For GPU arrays and operations
using CUDA.CUSPARSE          # For GPU sparse matrix support
using Krylov, LinearOperators # For advanced iterative solvers
using ..Element

export solve_system_gpu

# -----------------------------------------------------------------------
# GPU Sparse Solver (Assemble K, then solve on GPU)
# -----------------------------------------------------------------------
function get_free_dofs(bc_indicator::Matrix{T}) where T
    nNodes = size(bc_indicator, 1)
    ndof   = nNodes * 3
    constrained = falses(ndof)
    for i in 1:nNodes, j in 1:3
        if bc_indicator[i,j] > 0
            constrained[3*(i-1)+j] = true
        end
    end
    return findall(!, constrained)
end

function assemble_sparse_matrix(nodes::Matrix{T}, elements::Matrix{Int},
                                E::T, nu::T, density::Vector{T}) where T
    nNodes = size(nodes, 1)
    ndof   = nNodes * 3
    nElem  = size(elements, 1)

    I = Int[]
    J = Int[]
    V = T[]

    for e in 1:nElem
        conn = elements[e, :]
        elem_nodes = nodes[conn, :]

        E_local = E * density[e]
        ke = Element.hex_element_stiffness(elem_nodes, E_local, nu)

        for a in 1:24
            i_node = conn[div(a-1, 3)+1]
            i_dof  = 3*(i_node-1) + mod(a-1,3) + 1
            for b in 1:24
                j_node = conn[div(b-1, 3)+1]
                j_dof  = 3*(j_node-1) + mod(b-1,3) + 1
                push!(I, i_dof)
                push!(J, j_dof)
                push!(V, ke[a,b])
            end
        end
    end

    return sparse(I, J, V, ndof, ndof)
end

# -----------------------------------------------------------------------
# Native GPU Solver (using CUDA's built-in operations)
# -----------------------------------------------------------------------
function gpu_sparse_cg_solve(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                             bc_indicator::Matrix{T}, f::Vector{T},
                             density::Vector{T};
                             max_iter=1000, tol=1e-6) where T
    println("Assembling global sparse stiffness matrix (GPU path)...")
    K = assemble_sparse_matrix(nodes, elements, E, nu, density)
    ndof = size(K, 1)

    free_dofs = get_free_dofs(bc_indicator)
    K_free = K[free_dofs, free_dofs]
    f_free = f[free_dofs]

    println("Transferring system to GPU...")
    A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(K_free)
    b_gpu = CuVector(f_free)
    x_gpu = CuVector(zeros(T, length(f_free)))

    r_gpu = b_gpu - A_gpu * x_gpu
    p_gpu = copy(r_gpu)
    rsold_gpu = dot(r_gpu, r_gpu)

    println("Starting GPU-accelerated CG solver with $(length(free_dofs)) unknowns...")
    for iter in 1:max_iter
        Ap_gpu = A_gpu * p_gpu
        alpha = rsold_gpu / dot(p_gpu, Ap_gpu)
        x_gpu .= x_gpu .+ alpha .* p_gpu
        r_gpu .= r_gpu .- alpha .* Ap_gpu
        rsnew_gpu = dot(r_gpu, r_gpu)
        if sqrt(rsnew_gpu) < tol
            println("GPU CG converged after $iter iterations with residual $(sqrt(rsnew_gpu))")
            break
        end
        p_gpu .= r_gpu .+ (rsnew_gpu / rsold_gpu) .* p_gpu
        rsold_gpu = rsnew_gpu
        if iter % 100 == 0
            println("Iteration $iter, GPU residual: $(sqrt(rsnew_gpu))")
        end
    end

    x_free = Array(x_gpu)
    x_full = zeros(T, ndof)
    x_full[free_dofs] = x_free
    return x_full
end

# -----------------------------------------------------------------------
# Krylov.jl GPU Solver with IC(0) Preconditioning
# -----------------------------------------------------------------------
function gpu_krylov_solve(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                         bc_indicator::Matrix{T}, f::Vector{T},
                         density::Vector{T};
                         solver=:cg, max_iter=1000, tol=1e-6, use_precond=true) where T
    println("Assembling global sparse stiffness matrix (Krylov.jl GPU path)...")
    K = assemble_sparse_matrix(nodes, elements, E, nu, density)
    ndof = size(K, 1)

    free_dofs = get_free_dofs(bc_indicator)
    K_free = K[free_dofs, free_dofs]
    f_free = f[free_dofs]

    println("Transferring system to GPU...")
    A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(K_free)
    b_gpu = CuVector(f_free)
    
    # Setup preconditioner if requested
    if use_precond
        println("Computing IC(0) preconditioner...")
        try
            # Compute IC(0) decomposition
            P = ic02(A_gpu)
            
            # Additional vector required for solving triangular systems
            n = length(b_gpu)
            z = CUDA.zeros(T, n)
            
            # Define function to apply preconditioner
            function ldiv_ic0!(P::CuSparseMatrixCSR, x, y, z)
                ldiv!(z, LowerTriangular(P), x)   # Forward substitution with L
                ldiv!(y, LowerTriangular(P)', z)  # Backward substitution with Lᴴ
                return y
            end
            
            # Create linear operator for preconditioner
            symmetric = true
            hermitian = true
            opM = LinearOperator(T, n, n, symmetric, hermitian, 
                               (y, x) -> ldiv_ic0!(P, x, y, z))
            
            println("Preconditioner ready.")
        catch e
            println("Warning: Could not create IC(0) preconditioner: $e")
            println("Falling back to unpreconditioned solve.")
            opM = nothing
            use_precond = false
        end
    else
        opM = nothing
    end

    # Choose solver based on input
    println("Starting GPU-accelerated Krylov solver with $(length(free_dofs)) unknowns...")
    
    solve_start = time()
    if solver == :cg
        x_gpu, stats = cg(A_gpu, b_gpu, M=use_precond ? opM : nothing, 
                          itmax=max_iter, rtol=tol, verbose=1)
    elseif solver == :minres
        x_gpu, stats = minres(A_gpu, b_gpu, M=use_precond ? opM : nothing, 
                              itmax=max_iter, rtol=tol, verbose=1)
    elseif solver == :bicgstab
        x_gpu, stats = bicgstab(A_gpu, b_gpu, M=use_precond ? opM : nothing, 
                               itmax=max_iter, rtol=tol, verbose=1)
    else
        error("Unknown Krylov solver: $solver. Use :cg, :minres, or :bicgstab.")
    end
    solve_end = time()
    
    # Print solver statistics
    println("Krylov.jl $solver GPU solver completed:")
    println("  Iterations: $(stats.niter)")
    println("  Converged: $(stats.solved)")
    println("  Final residual: $(stats.rNorm)")
    println("  Solution time: $(solve_end - solve_start) seconds")

    # Transfer solution back to CPU and reconstruct full solution
    x_free = Array(x_gpu)
    x_full = zeros(T, ndof)
    x_full[free_dofs] = x_free
    return x_full
end

"""
    solve_system_gpu(nodes, elements, E, nu, bc_indicator, f,
                     density; max_iter=1000, tol=1e-6, 
                     method=:native, solver=:cg, use_precond=true)

High-level interface for the GPU-accelerated sparse solvers.

# Arguments
- `method`: Solution method to use, either `:native` for CUDA's built-in operations or 
            `:krylov` to use Krylov.jl iterative solvers
- `solver`: When using `:krylov` method, specifies which algorithm to use:
            `:cg`, `:minres`, or `:bicgstab`
- `use_precond`: When using `:krylov` method, whether to use IC(0) preconditioning
"""
function solve_system_gpu(nodes::Matrix{T}, elements::Matrix{Int}, E::T, nu::T,
                          bc_indicator::Matrix{T}, f::Vector{T},
                          density::Vector{T};
                          max_iter=1000, tol=1e-6, 
                          method=:native, solver=:cg, use_precond=true) where T
    
    @assert density !== nothing "You must provide a density array for GPU solver."
    
    if !CUDA.functional()
        error("CUDA is not functional on this system. Make sure you have a compatible GPU and CUDA drivers installed.")
    end
    
    solve_start = time()
    
    if method == :native
        solution = gpu_sparse_cg_solve(nodes, elements, E, nu, bc_indicator, f, density,
                                       max_iter=max_iter, tol=tol)
    elseif method == :krylov
        solution = gpu_krylov_solve(nodes, elements, E, nu, bc_indicator, f, density,
                                   solver=solver, max_iter=max_iter, tol=tol, 
                                   use_precond=use_precond)
    else
        error("Unknown method: $method. Use :native or :krylov.")
    end
    
    solve_end = time()
    @printf("Total solution time (GPU - %s): %.6f sec\n", 
            method == :native ? "native CG" : "Krylov.$solver", 
            solve_end - solve_start)
    return solution
end

# -----------------------------------------------------------------------
# Helper function for IC(0) factorization of sparse matrices
# -----------------------------------------------------------------------
function ic02(A::CuSparseMatrixCSR{T}) where T
    # Check if matrix is SPD (for a real problem)
    n = size(A, 1)
    if n == 0
        return CUDA.CUSPARSE.CuSparseMatrixCSR(spzeros(T, 0, 0))
    end
    
    # Create cusparseSpMatDescr for A
    descrA = CUSPARSE.cusparseSpMatDescr(A)
    
    # Setup for buffer size calculation
    bufferSize = Ref{Csize_t}(0)
    # Using csric02_bufferSize to get required buffer size
    CUSPARSE.cusparseSpSV_bufferSize(
        CUSPARSE.handle(),
        CUSPARSE.CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUDA.ones(T, 1),
        descrA,
        descrA,
        descrA,
        T,
        CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT,
        bufferSize
    )
    
    # Allocate buffer
    buffer = CUDA.zeros(UInt8, bufferSize[])
    
    # Create a copy of A for factorization
    factorized = copy(A)
    descrL = CUSPARSE.cusparseSpMatDescr(factorized)
    
    # Perform incomplete Cholesky factorization
    info = CUSPARSE.cusparseCreateCsric02Info()
    CUSPARSE.cusparseXcsric02_analysis(
        CUSPARSE.handle(),
        n, nnz(A), 
        descrA,
        A.nzVal, 
        A.rowPtr, 
        A.colVal,
        info,
        CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL,
        buffer
    )
    
    # Perform the factorization
    CUSPARSE.cusparseXcsric02(
        CUSPARSE.handle(),
        n, nnz(A),
        descrA,
        factorized.nzVal,
        factorized.rowPtr,
        factorized.colVal,
        info,
        CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL,
        buffer
    )
    
    CUSPARSE.cusparseDestroyCsric02Info(info)
    
    return factorized
end

end  # module GPUSolver