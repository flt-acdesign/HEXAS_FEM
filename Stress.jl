###########################################
# FILE: Stress.jl
###########################################

module Stress

using LinearAlgebra
using ..Element
export compute_stress_field

"""
    compute_element_stress(element_nodes, element_disp, E, nu)

Computes the 3x3 stress tensor at the center of a hexahedral element via B*U => strain => stress.
"""
function compute_element_stress(element_nodes::Array{Float32,2},
                                element_disp::Array{Float32,1},
                                E::Float32, nu::Float32)
    D = Element.material_matrix(E, nu)
    # Evaluate at center (xi=eta=zeta=0)
    xi, eta, zeta = 0.0f0, 0.0f0, 0.0f0
    _, dN_dxi = Element.shape_functions(xi, eta, zeta)
    J = transpose(dN_dxi)*element_nodes
    detJ = det(J)
    if detJ <= 0.0f0
        error("Non-positive Jacobian!")
    end
    invJ = inv(J)
    dN_dx = dN_dxi * transpose(invJ)

    # Build B
    B = zeros(Float32, 6, 24)
    for i in 1:8
        idx = 3*(i-1)+1
        dN_i = dN_dx[i, :]

        # Normal strain
        B[1, idx]   = dN_i[1]
        B[2, idx+1] = dN_i[2]
        B[3, idx+2] = dN_i[3]

        # Shear strain
        B[4, idx]   = dN_i[2]  # gamma_xy
        B[4, idx+1] = dN_i[1]
        B[5, idx+1] = dN_i[3]  # gamma_yz
        B[5, idx+2] = dN_i[2]
        B[6, idx]   = dN_i[3]  # gamma_xz
        B[6, idx+2] = dN_i[1]
    end

    strain = B * element_disp
    stress_voigt = D * strain

    σ = zeros(Float32, 3, 3)
    σ[1,1] = stress_voigt[1]  # σxx
    σ[2,2] = stress_voigt[2]  # σyy
    σ[3,3] = stress_voigt[3]  # σzz
    σ[1,2] = stress_voigt[4]; σ[2,1] = stress_voigt[4]  # τxy
    σ[2,3] = stress_voigt[5]; σ[3,2] = stress_voigt[5]  # τyz
    σ[1,3] = stress_voigt[6]; σ[3,1] = stress_voigt[6]  # τxz
    return σ
end

"""
    compute_principal_and_vonmises(σ)

Given a 3x3 stress tensor, returns (principal_stresses, von_mises).
Principal are sorted descending. Von Mises uses standard formula.
"""
function compute_principal_and_vonmises(σ::Matrix{Float32})
    eigvals = eigen(σ).values
    # Sort descending
    principal_stresses = sort(eigvals, rev=true)

    σxx = σ[1,1]
    σyy = σ[2,2]
    σzz = σ[3,3]
    σxy = σ[1,2]
    σyz = σ[2,3]
    σxz = σ[1,3]

    vm = sqrt(0.5f0 * ((σxx-σyy)^2 + (σyy-σzz)^2 + (σzz-σxx)^2) +
              3f0*(σxy^2 + σyz^2 + σxz^2))

    return principal_stresses, vm
end

"""
    compute_stress_field(nodes, elements, U, E, nu)

Loop over elements, compute:
  - principal stresses (3 x nElem)
  - von Mises (1 x nElem)
  - full stress in Voigt form (6 x nElem)

Returns (principal_field, vonmises_field, full_stress_voigt).
"""
function compute_stress_field(nodes, elements, U, E::Float32, nu::Float32)
    nElem = size(elements, 1)
    principal_field   = zeros(Float32, 3, nElem)
    vonmises_field    = zeros(Float32, nElem)
    full_stress_voigt = zeros(Float32, 6, nElem)

    for e in 1:nElem
        conn = elements[e, :]
        element_nodes = nodes[conn, :]

        # Gather element displacements
        element_disp = zeros(Float32, 24)
        for i in 1:8
            global_node = conn[i]
            element_disp[3*(i-1)+1 : 3*i] = U[3*(global_node-1)+1 : 3*global_node]
        end

        σ = compute_element_stress(element_nodes, element_disp, E, nu)
        (principal, vm) = compute_principal_and_vonmises(σ)

        principal_field[:, e] = principal
        vonmises_field[e]     = vm

        # (σxx, σyy, σzz, σxy, σyz, σxz)
        full_stress_voigt[:, e] .= (σ[1,1], σ[2,2], σ[3,3], σ[1,2], σ[2,3], σ[1,3])
    end

    return principal_field, vonmises_field, full_stress_voigt
end

end  # module Stress
