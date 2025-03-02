module MeshUtilities

export inside_sphere, inside_box,
       check_element_quality, fix_inverted_elements!,
       calculate_element_quality

using ..Element   # or using Main.HEXA.Element

using LinearAlgebra

"""
    inside_sphere(pt, center, diam)
Return true if point `pt` is inside a sphere of diameter `diam` at `center`.
"""
function inside_sphere(pt::AbstractVector, center::Tuple{Float32,Float32,Float32}, diam::Float32)
    r = diam / 2f0
    return norm(pt .- collect(center)) <= r
end

"""
    inside_box(pt, center, side)
Return true if point `pt` is inside a cube of side `side` centered at `center`.
"""
function inside_box(pt::AbstractVector, center::Tuple{Float32,Float32,Float32}, side::Float32)
    half = side / 2f0
    return abs(pt[1] - center[1]) <= half &&
           abs(pt[2] - center[2]) <= half &&
           abs(pt[3] - center[3]) <= half
end

"""
    check_element_quality(nodes, elements) -> poor_elements
Mark which elements are degenerate, etc.
"""
function check_element_quality(nodes::Matrix{Float32}, elements::Matrix{Int})
    nElem = size(elements,1)
    poor_elements = Int[]
    for e in 1:nElem
        # (Implementation you showed previously)
        # ...
    end
    return poor_elements
end

"""
    fix_inverted_elements!(nodes, elements) -> (fixed_count, warning_count)
Swap node ordering to fix negative Jacobians.
"""
function fix_inverted_elements!(nodes::Matrix{Float32}, elements::Matrix{Int})
    # ...
    return (fixed_count, warning_count)
end

"""
    calculate_element_quality(nodes, elements)
Returns (aspect_ratios, min_jacobians)
"""
function calculate_element_quality(nodes::Matrix{Float32}, elements::Matrix{Int})
    # ...
    return aspect_ratios, min_jacobians
end

end # module MeshUtilities
