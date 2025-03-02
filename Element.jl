###########################################
# FILE: Element.jl
###########################################

module Element

using LinearAlgebra
export NAT_COORDS, shape_functions, material_matrix, hex_element_stiffness

# Natural coordinates of the 8 vertices of a hexahedral element in [-1,1]^3
const NAT_COORDS = Float32[
    -1 -1 -1;
     1 -1 -1;
     1  1 -1;
    -1  1 -1;
    -1 -1  1;
     1 -1  1;
     1  1  1;
    -1  1  1
]

"""
    shape_functions(xi, eta, zeta)

Computes the trilinear shape functions and their derivatives at (xi, eta, zeta).
Returns (N, dN) with N=8 shape values, dN=8x3 derivative matrix.
"""
function shape_functions(xi, eta, zeta)
    N  = zeros(Float32, 8)
    dN = zeros(Float32, 8, 3)
    for i in 1:8
        xi_i, eta_i, zeta_i = NAT_COORDS[i,1], NAT_COORDS[i,2], NAT_COORDS[i,3]
        N[i] = 0.125f0*(1.0f0+xi*xi_i)*(1.0f0+eta*eta_i)*(1.0f0+zeta*zeta_i)
        dN[i,1] = 0.125f0 * xi_i*(1.0f0+eta*eta_i)*(1.0f0+zeta*zeta_i)
        dN[i,2] = 0.125f0*(1.0f0+xi*xi_i)*eta_i*(1.0f0+zeta*zeta_i)
        dN[i,3] = 0.125f0*(1.0f0+xi*xi_i)*(1.0f0+eta*eta_i)*zeta_i
    end
    return N, dN
end

"""
    material_matrix(E, nu)

Constructs the 6x6 isotropic material matrix for 3D elasticity.
"""
function material_matrix(E::Float32, nu::Float32)
    factor = E/((1.0f0+nu)*(1.0f0-2.0f0*nu))
    D = factor * Float32[
        1.0f0-nu  nu        nu        0.0f0 0.0f0 0.0f0;
        nu      1.0f0-nu    nu        0.0f0 0.0f0 0.0f0;
        nu       nu        1.0f0-nu   0.0f0 0.0f0 0.0f0;
        0.0f0    0.0f0     0.0f0    (1.0f0-2.0f0*nu)/2.0f0 0.0f0 0.0f0;
        0.0f0    0.0f0     0.0f0     0.0f0 (1.0f0-2.0f0*nu)/2.0f0 0.0f0;
        0.0f0    0.0f0     0.0f0     0.0f0 0.0f0 (1.0f0-2.0f0*nu)/2.0f0
    ]
    return D
end

"""
    hex_element_stiffness(nodes, E, nu)

Computes the 24x24 stiffness for a hex element via 2x2x2 Gauss integration.
"""
function hex_element_stiffness(nodes::AbstractMatrix{Float32}, E::Float32, nu::Float32)
    D = material_matrix(E, nu)
    ke = zeros(Float32, 24, 24)

    a = 1.0f0/sqrt(3.0f0)
    gauss_pts = Float32[-a, a]
    weights   = Float32[1.0f0, 1.0f0]

    for xi in gauss_pts, eta in gauss_pts, zeta in gauss_pts
        _, dN_dxi = shape_functions(xi, eta, zeta)
        J   = transpose(dN_dxi)*nodes
        detJ = det(J)
        if detJ <= 0.0f0
            error("Non-positive Jacobian!")
        end
        invJ = inv(J)
        dN_dx = dN_dxi * transpose(invJ)

        B = zeros(Float32, 6, 24)
        for i in 1:8
            idx = 3*(i-1)+1
            dN_i = dN_dx[i,:]

            # Normal strains
            B[1, idx]   = dN_i[1]
            B[2, idx+1] = dN_i[2]
            B[3, idx+2] = dN_i[3]

            # Shear strains
            B[4, idx]   = dN_i[2]  # gamma_xy
            B[4, idx+1] = dN_i[1]
            B[5, idx+1] = dN_i[3]  # gamma_yz
            B[5, idx+2] = dN_i[2]
            B[6, idx]   = dN_i[3]  # gamma_xz
            B[6, idx+2] = dN_i[1]
        end

        w = 1.0f0 # product of weights in each dimension = 1*1*1
        ke += transpose(B)*D*B * detJ*w
    end

    return ke
end

end  # module Element
