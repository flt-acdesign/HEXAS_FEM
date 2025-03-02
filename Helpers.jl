###########################################
# FILE: Helpers.jl
###########################################

module Helpers

using CUDA

export expand_element_indices, nodes_from_location, parse_location_component
export calculate_element_distribution, has_enough_gpu_memory, clear_gpu_memory

function expand_element_indices(elem_inds, dims)
    nElem_x = dims[1] - 1
    nElem_y = dims[2] - 1
    nElem_z = dims[3] - 1
    inds = Vector{Vector{Int}}()
    for d in 1:3
        if (typeof(elem_inds[d]) == String && elem_inds[d] == ":")
            if d == 1
                push!(inds, collect(1:nElem_x))
            elseif d == 2
                push!(inds, collect(1:nElem_y))
            elseif d == 3
                push!(inds, collect(1:nElem_z))
            end
        else
            push!(inds, [Int(elem_inds[d])])
        end
    end
    result = Int[]
    for i in inds[1], j in inds[2], k in inds[3]
        eidx = i + (j-1)*nElem_x + (k-1)*nElem_x*nElem_y
        push!(result, eidx)
    end
    return result
end

"""
    nodes_from_location(loc, dims)

Determines node indices from a non‑dimensional location specification.
loc: 3-element vector, each can be a number, ":" or "X%".
dims: (nNodes_x, nNodes_y, nNodes_z).
"""
function nodes_from_location(loc::Vector, dims)
    nNodes_x, nNodes_y, nNodes_z = dims
    ix = parse_location_component(loc[1], nNodes_x)
    iy = parse_location_component(loc[2], nNodes_y)
    iz = parse_location_component(loc[3], nNodes_z)
    nodes = Int[]
    for k in iz, j in iy, i in ix
        node = i + (j-1)*nNodes_x + (k-1)*nNodes_x*nNodes_y
        push!(nodes, node)
    end
    return nodes
end

function parse_location_component(val, nNodes::Int)
    if val == ":"
        return collect(1:nNodes)
    elseif isa(val, String) && endswith(val, "%")
        perc = parse(Float64, replace(val, "%"=>"")) / 100.0
        idx = round(Int, 1 + perc*(nNodes-1))
        return [idx]
    elseif isa(val, Number)
        if 0.0 <= val <= 1.0
            idx = round(Int, 1 + val*(nNodes-1))
            return [idx]
        else
            idx = clamp(round(Int, val), 1, nNodes)
            return [idx]
        end
    else
        error("Invalid location component: $val")
    end
end

# -------------------------------------------------------------
# GPU memory check
# -------------------------------------------------------------
function clear_gpu_memory()
    if !CUDA.functional()
        println("No GPU available.")
        return (0, 0)
    end
    GC.gc()
    CUDA.reclaim()

    final_free, total = CUDA.available_memory(), CUDA.total_memory()
    return (final_free, total)
end

function estimate_gpu_memory_required(nNodes, nElem)
    ndof = nNodes * 3
    nnz_estimate = ndof * 27
    # Each non-zero ~ 8 bytes for index + 8 for value => 16 bytes, approximate
    sparse_matrix_mem = nnz_estimate * (8 + 8)
    vectors_mem = ndof * 8 * 10
    element_matrices_mem = nElem * 24 * 24 * 8
    buffer_mem = (sparse_matrix_mem + vectors_mem) * 0.5
    total_mem = sparse_matrix_mem + vectors_mem + element_matrices_mem + buffer_mem
    return total_mem
end

function has_enough_gpu_memory(nNodes, nElem)
    if !CUDA.functional()
        return false
    end
    try
        free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory()
        required_mem = estimate_gpu_memory_required(nNodes, nElem)
        usable_mem = free_mem * 0.95
        println("Estimated GPU memory required: $(round(required_mem/1024^3, digits=2)) GB")
        println("Available GPU memory: $(round(free_mem/1024^3, digits=2)) GB (Total: $(round(total_mem/1024^3, digits=2)) GB)")
        println("Using $(round(required_mem/free_mem*100, digits=2))% of available GPU memory")
        has_enough = required_mem < usable_mem
        println("Has enough GPU memory: $(has_enough)")
        return has_enough
    catch e
        println("Error checking GPU memory: $e")
        return false
    end
end

function calculate_element_distribution(length_x, length_y, length_z, target_elem_count)
    total_volume = length_x * length_y * length_z
    # scale each dimension in proportion to the cubic root
    ratio_x = length_x / cbrt(total_volume)
    ratio_y = length_y / cbrt(total_volume)
    ratio_z = length_z / cbrt(total_volume)

    base_count = cbrt(target_elem_count)
    nElem_x = max(1, round(Int, base_count * ratio_x))
    nElem_y = max(1, round(Int, base_count * ratio_y))
    nElem_z = max(1, round(Int, base_count * ratio_z))

    dx = length_x / nElem_x
    dy = length_y / nElem_y
    dz = length_z / nElem_z
    actual_elem_count = nElem_x * nElem_y * nElem_z
    return nElem_x, nElem_y, nElem_z, Float32(dx), Float32(dy), Float32(dz), actual_elem_count
end

end  # module Helpers
