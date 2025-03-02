module Mesh

export node_index, generate_mesh

using LinearAlgebra, Printf

"""
    node_index(i, j, k, nNodes_x, nNodes_y)

Converts 3D indices (i, j, k) into a linear node index (column‑major ordering).
"""
function node_index(i, j, k, nNodes_x, nNodes_y)
    return i + (j-1)*nNodes_x + (k-1)*nNodes_x*nNodes_y
end

"""
    element_centroid(e, nodes, elements)

Computes the centroid of element `e` given the node coordinates.
"""
function element_centroid(e, nodes, elements)
    conn = elements[e, :]
    elem_nodes = nodes[conn, :]
    c = zeros(Float32, 3)
    for i in 1:8
        c .+= elem_nodes[i, :]
    end
    return c ./ 8.0f0
end

# Helper functions for shape testing
function inside_sphere(pt::Vector{Float32}, center::Tuple{Float32,Float32,Float32}, diam::Float32)
    r = diam / 2.0f0
    return norm(pt .- collect(center)) <= r
end

function inside_box(pt::Vector{Float32}, center::Tuple{Float32,Float32,Float32}, side::Float32)
    half = side / 2.0f0
    return abs(pt[1] - center[1]) <= half &&
           abs(pt[2] - center[2]) <= half &&
           abs(pt[3] - center[3]) <= half
end

"""
    generate_mesh(nElem_x, nElem_y, nElem_z;
                  dx=1.0f0, dy=1.0f0, dz=1.0f0,
                  shapes_to_add=[], shapes_to_remove=[],
                  remove_unused_nodes=true)

Generates a structured hexahedral mesh and then:
  1. Adds extra elements from shapes with action "add" (avoiding duplicates)
  2. Removes any element whose centroid lies inside a shape with action "remove"
  3. Optionally removes unused nodes and renumbers the connectivity

Returns (nodes, elements, dims).
"""
function generate_mesh(nElem_x::Int, nElem_y::Int, nElem_z::Int;
                       dx::Float32=1.0f0,
                       dy::Float32=1.0f0,
                       dz::Float32=1.0f0,
                       shapes_to_add = Any[],
                       shapes_to_remove = Any[],
                       remove_unused_nodes::Bool=true)
    # ---------------------------
    # 1. Generate the base (structured) mesh.
    # ---------------------------
    nNodes_x = nElem_x + 1
    nNodes_y = nElem_y + 1
    nNodes_z = nElem_z + 1
    dims = (nNodes_x, nNodes_y, nNodes_z)
    
    nNodes = nNodes_x * nNodes_y * nNodes_z
    base_nodes = zeros(Float32, nNodes, 3)
    idx = 1
    for k in 1:nNodes_z, j in 1:nNodes_y, i in 1:nNodes_x
        base_nodes[idx, :] = [(i-1)*dx, (j-1)*dy, (k-1)*dz]
        idx += 1
    end
    
    nElem = (nNodes_x - 1) * (nNodes_y - 1) * (nNodes_z - 1)
    base_elements = Matrix{Int}(undef, nElem, 8)
    elem_idx = 1
    for k in 1:(nNodes_z-1), j in 1:(nNodes_y-1), i in 1:(nNodes_x-1)
        n1 = node_index(i, j, k, nNodes_x, nNodes_y)
        n2 = node_index(i+1, j, k, nNodes_x, nNodes_y)
        n3 = node_index(i+1, j+1, k, nNodes_x, nNodes_y)
        n4 = node_index(i, j+1, k, nNodes_x, nNodes_y)
        n5 = node_index(i, j, k+1, nNodes_x, nNodes_y)
        n6 = node_index(i+1, j, k+1, nNodes_x, nNodes_y)
        n7 = node_index(i+1, j+1, k+1, nNodes_x, nNodes_y)
        n8 = node_index(i, j+1, k+1, nNodes_x, nNodes_y)
        base_elements[elem_idx, :] = [n1, n2, n3, n4, n5, n6, n7, n8]
        elem_idx += 1
    end

    # ---------------------------
    # 2. Process "add" shapes to generate extra elements.
    # ---------------------------
    all_new_nodes = Vector{Vector{Float32}}()  # new nodes (each a 3-element vector)
    all_new_elements = Vector{Vector{Int}}()
    
    # Use a set to track extra elements (using sorted connectivity as a key) to avoid duplicates.
    new_elem_set = Set{NTuple{8,Int}}()
    
    for shape in shapes_to_add
        shape_type = lowercase(shape["type"])
        if shape_type == "sphere"
            center = tuple(Float32.(shape["center"])...)
            diam = Float32(shape["diameter"])
            sphere_radius = diam / 2.0f0
            sx = center[1] - sphere_radius; ex = center[1] + sphere_radius
            sy = center[2] - sphere_radius; ey = center[2] + sphere_radius
            sz = center[3] - sphere_radius; ez = center[3] + sphere_radius
            sx_new = floor(sx / dx) * dx; ex_new = ceil(ex / dx) * dx
            sy_new = floor(sy / dy) * dy; ey_new = ceil(ey / dy) * dy
            sz_new = floor(sz / dz) * dz; ez_new = ceil(ez / dz) * dz
            nx = round(Int, (ex_new - sx_new) / dx)
            ny = round(Int, (ey_new - sy_new) / dy)
            nz = round(Int, (ez_new - sz_new) / dz)
            println("  Sphere add grid: $(nx) x $(ny) x $(nz) cells")
            
            # Build a local grid and map nodes.
            sphere_node_map = Dict{Tuple{Int,Int,Int}, Int}()
            next_node_id = size(base_nodes, 1) + length(all_new_nodes) + 1
            for k in 0:nz, j in 0:ny, i in 0:nx
                x = sx_new + i*dx
                y = sy_new + j*dy
                z = sz_new + k*dz
                if sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2) <= sphere_radius * 1.01f0
                    # Reuse node if within base mesh.
                    length_x = dx * (nNodes_x - 1)
                    length_y = dy * (nNodes_y - 1)
                    length_z = dz * (nNodes_z - 1)
                    if 0.0f0 <= x <= length_x && 0.0f0 <= y <= length_y && 0.0f0 <= z <= length_z
                        i_basic = round(Int, x/dx) + 1
                        j_basic = round(Int, y/dy) + 1
                        k_basic = round(Int, z/dz) + 1
                        node_idx = node_index(i_basic, j_basic, k_basic, nNodes_x, nNodes_y)
                        sphere_node_map[(i,j,k)] = node_idx
                    else
                        sphere_node_map[(i,j,k)] = next_node_id
                        push!(all_new_nodes, [x, y, z])
                        next_node_id += 1
                    end
                end
            end
            
            for k in 0:nz-1, j in 0:ny-1, i in 0:nx-1
                corners = (
                    get(sphere_node_map, (i, j, k), nothing),
                    get(sphere_node_map, (i+1, j, k), nothing),
                    get(sphere_node_map, (i+1, j+1, k), nothing),
                    get(sphere_node_map, (i, j+1, k), nothing),
                    get(sphere_node_map, (i, j, k+1), nothing),
                    get(sphere_node_map, (i+1, j, k+1), nothing),
                    get(sphere_node_map, (i+1, j+1, k+1), nothing),
                    get(sphere_node_map, (i, j+1, k+1), nothing)
                )
                if any(x -> x === nothing, corners)
                    continue
                end
                # Compute the element centroid using a reduction.
                pts = [ if idx ≤ size(base_nodes,1)
                            base_nodes[idx, :]
                        else
                            all_new_nodes[idx - size(base_nodes,1)]
                        end for idx in corners ]
                centroid = reduce(+, pts) ./ 8.0f0
                if inside_sphere(centroid, center, diam)
                    key = ntuple(i -> sort(collect(corners))[i], 8)
                    if key ∉ new_elem_set
                        push!(new_elem_set, key)
                        push!(all_new_elements, collect(corners))
                    end
                end
            end
            
        elseif shape_type == "box"
            center = tuple(Float32.(shape["center"])...)
            side = Float32(shape["side"])
            half_box = side / 2.0f0
            bx = center[1] - half_box; ex = center[1] + half_box
            by_ = center[2] - half_box; ey_ = center[2] + half_box
            bz = center[3] - half_box; ez = center[3] + half_box
            bx_new = floor(bx / dx) * dx; ex_new = ceil(ex / dx) * dx
            by_new = floor(by_ / dy) * dy; ey_new = ceil(ey_ / dy) * dy
            bz_new = floor(bz / dz) * dz; ez_new = ceil(ez / dz) * dz
            nx = round(Int, (ex_new - bx_new) / dx)
            ny = round(Int, (ey_new - by_new) / dy)
            nz = round(Int, (ez_new - bz_new) / dz)
            println("  Box add grid: $(nx) x $(ny) x $(nz) cells")
            
            box_node_map = Dict{Tuple{Int,Int,Int}, Int}()
            next_node_id = size(base_nodes, 1) + length(all_new_nodes) + 1
            for k in 0:nz, j in 0:ny, i in 0:nx
                x = bx_new + i*dx
                y = by_new + j*dy
                z = bz_new + k*dz
                if (x >= center[1] - half_box*1.01f0 && x <= center[1] + half_box*1.01f0) &&
                   (y >= center[2] - half_box*1.01f0 && y <= center[2] + half_box*1.01f0) &&
                   (z >= center[3] - half_box*1.01f0 && z <= center[3] + half_box*1.01f0)
                    if 0.0f0 <= x <= dx*(nNodes_x-1) && 0.0f0 <= y <= dy*(nNodes_y-1) && 0.0f0 <= z <= dz*(nNodes_z-1)
                        i_basic = round(Int, x/dx) + 1
                        j_basic = round(Int, y/dy) + 1
                        k_basic = round(Int, z/dz) + 1
                        node_idx = node_index(i_basic, j_basic, k_basic, nNodes_x, nNodes_y)
                        box_node_map[(i,j,k)] = node_idx
                    else
                        box_node_map[(i,j,k)] = next_node_id
                        push!(all_new_nodes, [x, y, z])
                        next_node_id += 1
                    end
                end
            end
            for k in 0:nz-1, j in 0:ny-1, i in 0:nx-1
                corners = (
                    get(box_node_map, (i, j, k), nothing),
                    get(box_node_map, (i+1, j, k), nothing),
                    get(box_node_map, (i+1, j+1, k), nothing),
                    get(box_node_map, (i, j+1, k), nothing),
                    get(box_node_map, (i, j, k+1), nothing),
                    get(box_node_map, (i+1, j, k+1), nothing),
                    get(box_node_map, (i+1, j+1, k+1), nothing),
                    get(box_node_map, (i, j+1, k+1), nothing)
                )
                if any(x -> x === nothing, corners)
                    continue
                end
                pts = [ if idx ≤ size(base_nodes,1)
                            base_nodes[idx, :]
                        else
                            all_new_nodes[idx - size(base_nodes,1)]
                        end for idx in corners ]
                centroid = reduce(+, pts) ./ 8.0f0
                if inside_box(centroid, center, side)
                    key = ntuple(i -> sort(collect(corners))[i], 8)
                    if key ∉ new_elem_set
                        push!(new_elem_set, key)
                        push!(all_new_elements, collect(corners))
                    end
                end
            end
            
        else
            @warn "Unknown shape type in addition: $shape_type"
        end
    end

    # ---------------------------
    # 3. Combine base elements with added elements, remove duplicates, and process removal shapes.
    # ---------------------------
    union_elements = vcat([vec(base_elements[i, :]) for i in 1:size(base_elements, 1)], all_new_elements)
    
    unique_elems = Vector{Vector{Int}}()
    seen = Set{NTuple{8,Int}}()
    for elem in union_elements
        key = ntuple(i -> elem[i], 8)
        if key ∉ seen
            push!(seen, key)
            push!(unique_elems, elem)
        end
    end

    final_elements = Vector{Vector{Int}}()
    for elem in unique_elems
        pts = [ (idx ≤ size(base_nodes,1) ? base_nodes[idx, :] : all_new_nodes[idx - size(base_nodes,1)]) for idx in elem ]
        centroid = reduce(+, pts) ./ 8.0f0
        remove = false
        for shape in shapes_to_remove
            shape_type = lowercase(shape["type"])
            if shape_type == "sphere"
                center = tuple(Float32.(shape["center"])...)
                diam = Float32(shape["diameter"])
                if inside_sphere(centroid, center, diam)
                    remove = true
                    break
                end
            elseif shape_type == "box"
                center = tuple(Float32.(shape["center"])...)
                side = Float32(shape["side"])
                if inside_box(centroid, center, side)
                    remove = true
                    break
                end
            end
        end
        if !remove
            push!(final_elements, elem)
        end
    end

    # ---------------------------
    # 4. Reconstruct the full node list and renumber element connectivity.
    # ---------------------------
    nodes_all = vcat(base_nodes, isempty(all_new_nodes) ? Array{Float32}(undef, 0, 3) : reduce(vcat, [reshape(n, 1, 3) for n in all_new_nodes]))
    nNodes_total = size(nodes_all, 1)
    used = falses(nNodes_total)
    for elem in final_elements
        for n in elem
            used[n] = true
        end
    end
    old_to_new = cumsum(used)
    new_nNodes = sum(used)
    new_nodes = zeros(Float32, new_nNodes, 3)
    for i in 1:nNodes_total
        if used[i]
            new_nodes[old_to_new[i], :] = nodes_all[i, :]
        end
    end
    for i in 1:length(final_elements)
        for j in 1:8
            final_elements[i][j] = old_to_new[ final_elements[i][j] ]
        end
    end
    
    println("Element processing summary:")
    println("  - Base elements: $(size(base_elements, 1))")
    println("  - Added extra elements: $(length(all_new_elements))")
    println("  - Final mesh elements after removal: $(length(final_elements))")
    println("Node processing:")
    println("  - Original nodes: $(nNodes_total)")
    println("  - Used nodes: $(new_nNodes)")
    println("  - Removed nodes: $(nNodes_total - new_nNodes)")
    
    return new_nodes, reduce(vcat, [reshape(e, 1, 8) for e in final_elements]), dims
end

end  # module Mesh
