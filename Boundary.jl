###########################################
# FILE: Boundary.jl
###########################################

module Boundary

using JSON
using SparseArrays

# Export the main boundary functions
export get_bc_indicator, reduce_system, apply_external_forces!

"""
    get_affected_nodes(spec, nodes)

Returns an array of *final* node indices affected by this BC specification `spec`.
`spec` can have:
  - "node" => a direct 1-based node index (or array of indices)
  - "location" => a 3-element array describing x, y, z selection in either absolute coords,
                  fractions [0..1], or the wildcard ":".

For example:
  - "location": [0.0, ":", 1.0]
    means "all nodes whose x-coordinate is near 0.0, any y, and z near maximum"
"""
function get_affected_nodes(spec::Dict, nodes::Matrix{Float32})
    nNodes = size(nodes, 1)

    # 1) If user gave "node"
    if haskey(spec, "node")
        raw = spec["node"]
        if isa(raw, Integer)
            idx = clamp(raw, 1, nNodes)
            return [idx]
        elseif isa(raw, AbstractVector)
            node_list = Int[]
            for r in raw
                push!(node_list, clamp(r, 1, nNodes))
            end
            return unique(node_list)
        else
            error("'node' must be an integer or an array of integers")
        end
    end

    # 2) If user gave "location"
    if haskey(spec, "location")
        loc_array = spec["location"]
        if length(loc_array) < 3
            error("Location specification must have at least 3 components (x,y,z)")
        end
        return get_nodes_by_location(loc_array, nodes)
    end

    error("Specification must include either 'node' or 'location'")
end

"""
    get_nodes_by_location(loc_array, nodes)

Find nodes whose (x,y,z) coordinates match the "location" pattern.
Each component in loc_array can be:
  - a Number in [0..1], treated as a fraction of (min..max)
  - a Number outside [0..1], treated as an absolute coordinate (with tolerance)
  - a String like ":", meaning "all values"
  - a String like "50%", meaning fraction 0.50
"""
function get_nodes_by_location(loc_array, nodes::Matrix{Float32})
    xvals = nodes[:, 1]
    yvals = nodes[:, 2]
    zvals = nodes[:, 3]

    xmin, xmax = extrema(xvals)
    ymin, ymax = extrema(yvals)
    zmin, zmax = extrema(zvals)

    xspec = loc_array[1]
    yspec = loc_array[2]
    zspec = loc_array[3]

    xmask = interpret_location_component(xspec, xvals, xmin, xmax)
    ymask = interpret_location_component(yspec, yvals, ymin, ymax)
    zmask = interpret_location_component(zspec, zvals, zmin, zmax)

    return findall(i -> xmask[i] && ymask[i] && zmask[i], eachindex(xvals))
end

"""
    interpret_location_component(spec, coords, cmin, cmax)

Returns a Bool array indicating which nodes match 'spec' in this dimension.
No references to literals with f0 suffix; we use Float32 casting explicitly.
"""
function interpret_location_component(spec,
                                      coords::Vector{Float32},
                                      cmin::Float32, cmax::Float32)
    nNodes = length(coords)
    mask = falses(nNodes)

    if spec == ":"
        return trues(nNodes)
    elseif isa(spec, Number)
        # If 0 <= spec <= 1, treat as fraction; else absolute coordinate
        if spec >= 0.0 && spec <= 1.0
            val = Float32(cmin + spec*(cmax - cmin))
            tol = Float32(1e-6) * max(1.0f0, abs(cmax - cmin))
            @inbounds for i in 1:nNodes
                if abs(coords[i] - val) <= tol
                    mask[i] = true
                end
            end
        else
            val = Float32(spec)
            tol = Float32(1e-6) * max(1.0f0, abs(cmax - cmin))
            @inbounds for i in 1:nNodes
                if abs(coords[i] - val) <= tol
                    mask[i] = true
                end
            end
        end
    elseif isa(spec, String) && endswith(spec, "%")
        # e.g. "50%" => fraction 0.5
        frac = parse(Float32, replace(spec, "%"=>"")) / 100.0f0
        frac = clamp(frac, 0.0f0, 1.0f0)
        val = Float32(cmin + frac*(cmax - cmin))
        tol = Float32(1e-6) * max(1.0f0, abs(cmax - cmin))
        @inbounds for i in 1:nNodes
            if abs(coords[i] - val) <= tol
                mask[i] = true
            end
        end
    else
        error("Invalid location component: $spec")
    end

    return mask
end

"""
    get_bc_indicator(nNodes, nodes, bc_data; T=Float32)

Generates an (nNodes x 3) matrix with 1.0 where a boundary condition is applied,
and 0.0 otherwise. We look up nodes by coordinate, not dims.
"""
function get_bc_indicator(nNodes::Int,
                          nodes::Matrix{Float32},
                          bc_data::Vector{Any};  # <--- Accept Vector{Any} or Vector{<:AbstractDict}
                          T::Type{<:AbstractFloat} = Float32)

    bc_indicator = zeros(T, nNodes, 3)
    for bc in bc_data
        dofs = bc["DoFs"]
        for dof in dofs
            if dof < 1 || dof > 3
                error("Invalid DoF index: $dof (must be 1..3).")
            end
        end

        # find affected nodes
        affected = get_affected_nodes(bc, nodes)
        for nd in affected
            for d in dofs
                bc_indicator[nd, d] = one(T)
            end
        end
    end

    return bc_indicator
end

"""
    reduce_system(K, F, bc_data, nodes, elements)

Use bc_data + final node array to mark constrained DOFs,
then zero them out of the stiffness/force system.
"""
function reduce_system(K::SparseMatrixCSC{Float32,Int},
                      F::Vector{Float32},
                      bc_data::Vector{Any},   # or Vector{<:AbstractDict}
                      nodes::Matrix{Float32},
                      elements::Matrix{Int})

    nNodes = size(nodes, 1)
    ndof   = 3*nNodes
    constrained = falses(ndof)

    for bc in bc_data
        dofs = bc["DoFs"]
        affected = get_affected_nodes(bc, nodes)
        for nd in affected
            for d in dofs
                gdof = 3*(nd-1) + d
                constrained[gdof] = true
                F[gdof] = 0.0f0
            end
        end
    end

    free_indices = findall(!, constrained)
    K_reduced = K[free_indices, free_indices]
    F_reduced = F[free_indices]
    return K_reduced, F_reduced, free_indices
end

"""
    apply_external_forces!(F, forces_data, nodes, elements)

Look up affected nodes by coordinate or direct node index,
then apply the force. If "location" is given, spread total force
among all affected nodes equally.
"""
function apply_external_forces!(F::Vector{T},
    forces_data::Vector{Any},  # or Vector{<:AbstractDict}
    nodes::Matrix{Float32},
    elements::Matrix{Int}) where T<:AbstractFloat

for force in forces_data
affected_nodes = get_affected_nodes(force, nodes)

# -------------------------- FIX: Check for empty list of nodes --------------------------
if isempty(affected_nodes)
@warn "No nodes found for force specification; skipping this force: $(force)"
continue
end
# ----------------------------------------------------------------------------------------

f_arr = force["F"]
if length(f_arr) < 3
f_arr = vcat(f_arr, zeros(3 - length(f_arr)))
end
f_arr = T.(f_arr)

# If user gave "location", we spread the total force among the matched nodes
scale_factor = haskey(force, "location") ? 1 / length(affected_nodes) : one(T)

for nd in affected_nodes
for i in 1:3
global_dof = 3*(nd-1) + i
F[global_dof] += scale_factor * f_arr[i]
end
end
end

return F
end

end  # module Boundary
