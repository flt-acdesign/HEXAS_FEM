module MeshShapeProcessing

export process_add_shape!, process_remove_shape!

using LinearAlgebra
using ..Element           # if you need shape_functions or something
using ..MeshUtilities     # for inside_sphere, inside_box, etc.

"""
    process_add_shape!(shape, nodes, dx, dy, dz)
Generic “add” routine that returns new element connectivities 
(but does not modify your main `elements`). 
"""
function process_add_shape!(shape::Dict, nodes, dx, dy, dz)
    shape_type = lowercase(shape["type"])
    if shape_type == "sphere"
        return make_sphere_elements(shape, nodes, dx, dy, dz)
    elseif shape_type == "box"
        return make_box_elements(shape, nodes, dx, dy, dz)
    else
        @warn "Unknown shape type for 'add': $shape_type"
        return NTuple{8,Int}[]
    end
end

"""
    process_remove_shape!(shape, nodes, elements, remove_mask)
Mark `remove_mask[e] = true` if the element's centroid is inside shape.
"""
function process_remove_shape!(shape::Dict,
                               nodes::Matrix{Float32},
                               elements::Matrix{Int},
                               remove_mask::BitVector)
    shape_type = lowercase(shape["type"])
    if shape_type == "sphere"
        remove_sphere!(shape, nodes, elements, remove_mask)
    elseif shape_type == "box"
        remove_box!(shape, nodes, elements, remove_mask)
    else
        @warn "Unknown shape type for 'remove': $shape_type"
    end
end

############### Internal “Add” Routines ###############
function make_sphere_elements(shape, nodes, dx, dy, dz)
    if !haskey(shape,"center") || !haskey(shape,"diameter")
        @warn "Sphere shape missing 'center' or 'diameter'; skipping."
        return NTuple{8,Int}[]
    end
    @warn "make_sphere_elements does NOT generate new subdivided mesh so far."
    return NTuple{8,Int}[]
end

function make_box_elements(shape, nodes, dx, dy, dz)
    if !haskey(shape,"center") || !haskey(shape,"side")
        @warn "Box shape missing 'center' or 'side'; skipping."
        return NTuple{8,Int}[]
    end
    @warn "make_box_elements does NOT generate new subdivided mesh so far."
    return NTuple{8,Int}[]
end

############### Internal “Remove” Routines ###############
function remove_sphere!(shape::Dict,
                        nodes::Matrix{Float32},
                        elements::Matrix{Int},
                        remove_mask::BitVector)
    center = tuple(Float32.(shape["center"])...)
    diam   = Float32(shape["diameter"])
    for e in 1:size(elements,1)
        if !remove_mask[e]
            c = element_centroid(e, nodes, elements)
            if inside_sphere(c, center, diam)
                remove_mask[e] = true
            end
        end
    end
end

function remove_box!(shape::Dict,
                     nodes::Matrix{Float32},
                     elements::Matrix{Int},
                     remove_mask::BitVector)
    center = tuple(Float32.(shape["center"])...)
    side   = Float32(shape["side"])
    for e in 1:size(elements,1)
        if !remove_mask[e]
            c = element_centroid(e, nodes, elements)
            if inside_box(c, center, side)
                remove_mask[e] = true
            end
        end
    end
end

############### Helper ###############
function element_centroid(e::Int,
                          nodes::Matrix{Float32},
                          elements::Matrix{Int})
    conn = elements[e, :]
    elem_nodes = nodes[conn, :]
    centroid = sum(elem_nodes, dims=1) ./ 8.0f0
    return vec(centroid)
end

end # module MeshShapeProcessing
