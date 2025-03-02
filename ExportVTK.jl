module ExportVTK

using Printf

export export_mesh, export_solution

"""
    export_mesh(nodes, elements; bc_indicator=nothing, filename="mesh_output.vtu")

Exports a 3D hexahedral mesh to a legacy VTK file in ASCII format.
If `bc_indicator` is supplied, it can be exported as a point data field.

# Arguments
- `nodes`:  N×3 matrix of node coordinates
- `elements`: M×8 connectivity, each row lists the 8 node indices of a hexahedral element
- `bc_indicator`: (optional) an N×k matrix you want to store as node-based data
- `filename`: output path, defaults to "mesh_output.vtu"
"""
function export_mesh(nodes::Matrix{Float32},
                    elements::Matrix{Int};
                    bc_indicator=nothing,
                    filename::String="mesh_output.vtu")
    
    # Ensure the filename has an extension
    if !endswith(lowercase(filename), ".vtk") && !endswith(lowercase(filename), ".vtu")
        filename *= ".vtk"
    end
    
    # Sanitize node coordinates (replace NaN/Inf with zero)
    if any(isnan, nodes) || any(isinf, nodes)
        @warn "Found NaN or Inf values in node coordinates. Replacing with zeros."
        nodes = replace(nodes, NaN => 0.0f0, Inf => 0.0f0, -Inf => 0.0f0)
    end
    
    # Check for extreme values that might cause ParaView to crash
    max_coord = maximum(abs.(nodes))
    if max_coord > 1.0e10
        @warn "Very large coordinate values detected (maximum absolute value: $max_coord). Clamping to reasonable range."
        nodes = clamp.(nodes, -1.0e10f0, 1.0e10f0)
    end

    nElem  = size(elements, 1)
    nNodes = size(nodes, 1)

    # Validate connectivity and filter out invalid elements
    valid_elements = Int[]
    for e = 1:nElem
        elem_nodes = elements[e, :]
        # Check for invalid or out-of-bounds node indices
        if any(n -> n < 1 || n > nNodes, elem_nodes)
            @warn "Element $e has invalid node indices; skipping it entirely."
        else
            push!(valid_elements, e)
        end
    end

    # If no valid elements remain, bail out gracefully
    if isempty(valid_elements)
        @warn "No valid elements found. Skipping VTK export."
        return
    end

    nElem_valid = length(valid_elements)
    
    try
        open(filename, "w") do file
            # Write VTK header
            write(file, "# vtk DataFile Version 3.0\n")
            write(file, "HEXA FEM Mesh (ASCII)\n")
            write(file, "ASCII\n")
            write(file, "DATASET UNSTRUCTURED_GRID\n")
            
            # Write node coordinates
            write(file, "POINTS $(nNodes) float\n")
            for i in 1:nNodes
                @printf(file, "%.6f %.6f %.6f\n", nodes[i, 1], nodes[i, 2], nodes[i, 3])
            end
            
            # Write cell connectivity (VTK uses 0-based indexing)
            write(file, "\nCELLS $(nElem_valid) $(nElem_valid * 9)\n")
            for idx in valid_elements
                # First number is the count of points in the cell (8 for hex)
                @printf(file, "8 %d %d %d %d %d %d %d %d\n", 
                       elements[idx, 1]-1, elements[idx, 2]-1, elements[idx, 3]-1, elements[idx, 4]-1,
                       elements[idx, 5]-1, elements[idx, 6]-1, elements[idx, 7]-1, elements[idx, 8]-1)
            end
            
            # Write cell types (12 = VTK_HEXAHEDRON)
            write(file, "\nCELL_TYPES $(nElem_valid)\n")
            for _ in 1:nElem_valid
                write(file, "12\n")
            end
            
            # Write boundary condition data if provided
            if bc_indicator !== nothing && size(bc_indicator, 1) == nNodes
                write(file, "\nPOINT_DATA $(nNodes)\n")
                
                # Get the number of BC components
                ncols_bc = min(size(bc_indicator, 2), 3)
                
                if ncols_bc >= 2
                    # Export as vector field if we have multiple components
                    write(file, "VECTORS BC float\n")
                    for i in 1:nNodes
                        if ncols_bc == 2
                            @printf(file, "%.6f %.6f 0.0\n", 
                                  bc_indicator[i, 1], bc_indicator[i, 2])
                        else
                            @printf(file, "%.6f %.6f %.6f\n", 
                                  bc_indicator[i, 1], bc_indicator[i, 2], bc_indicator[i, 3])
                        end
                    end
                else
                    # Single component as scalar
                    write(file, "SCALARS BCx float 1\n")
                    write(file, "LOOKUP_TABLE default\n")
                    for i in 1:nNodes
                        @printf(file, "%.6f\n", bc_indicator[i, 1])
                    end
                end
            end
        end
        
        println("Successfully exported mesh to $filename (ASCII format)")
    catch e
        @error "Failed to save VTK file: $e"
        println("Error details: ", e)
    end
end

"""
    export_solution(nodes, elements, U_full, F, bc_indicator,
                   principal_field, vonmises_field, full_stress_voigt;
                   scale=1.0f0, filename="solution_output.vtu")

Exports the solution results to VTK files in ASCII format, creating separate files
for original and deformed meshes.
"""
function export_solution(nodes::Matrix{Float32},
                        elements::Matrix{Int},
                        U_full::Vector{Float32},
                        F::Vector{Float32},
                        bc_indicator::Matrix{Float32},
                        principal_field::Matrix{Float32},
                        vonmises_field::Vector{Float32},
                        full_stress_voigt::Matrix{Float32};
                        scale::Float32=1.0f0,
                        filename::String="solution_output.vtu")

    # Replace NaN/Inf in all input arrays
    function sanitize_data(data)
        # Replace any NaN/Inf values
        data = replace(data, NaN => 0.0f0, Inf => 0.0f0, -Inf => 0.0f0)
        
        # Clamp extreme values that might cause ParaView to crash
        max_val = maximum(abs.(data))
        if max_val > 1.0e10
            @warn "Very large values detected (max abs: $max_val). Clamping to prevent ParaView crashes."
            return clamp.(data, -1.0e10f0, 1.0e10f0)
        end
        return data
    end
    
    # Sanitize all input data
    U_full = sanitize_data(U_full)
    F = sanitize_data(F)
    nodes = sanitize_data(nodes)
    principal_field = sanitize_data(principal_field)
    vonmises_field = sanitize_data(vonmises_field)
    full_stress_voigt = sanitize_data(full_stress_voigt)

    nNodes = size(nodes, 1)
    nElem  = size(elements, 1)

    # Validate connectivity and filter out invalid elements
    valid_elements = Int[]
    for e = 1:nElem
        elem_nodes = elements[e, :]
        # Check for invalid or out-of-bounds node indices
        if any(n -> n < 1 || n > nNodes, elem_nodes)
            @warn "Element $e has invalid node indices; skipping it."
        else
            push!(valid_elements, e)
        end
    end

    nElem_valid = length(valid_elements)
    if nElem_valid == 0
        @warn "No valid elements remain. Skipping solution export."
        return
    end

    # Pad or truncate arrays to match the required sizes
    function ensure_array_size(arr, expected_size, pad_value=0.0f0)
        if length(arr) < expected_size
            return vcat(arr, fill(pad_value, expected_size - length(arr)))
        elseif length(arr) > expected_size
            return arr[1:expected_size]
        else
            return arr
        end
    end
    
    # Ensure correct array sizes for vector fields
    U_full = ensure_array_size(U_full, 3*nNodes)
    F = ensure_array_size(F, 3*nNodes)
    
    # Create displacement and force arrays in node-wise format
    displacement = zeros(Float32, nNodes, 3)
    forces = zeros(Float32, nNodes, 3)
    
    for i in 1:nNodes
        base_idx = 3*(i-1)
        if base_idx + 3 <= length(U_full)
            displacement[i, 1] = U_full[base_idx + 1]
            displacement[i, 2] = U_full[base_idx + 2]
            displacement[i, 3] = U_full[base_idx + 3]
        end
        
        if base_idx + 3 <= length(F)
            forces[i, 1] = F[base_idx + 1]
            forces[i, 2] = F[base_idx + 2]
            forces[i, 3] = F[base_idx + 3]
        end
    end
    
    # Displacement magnitude
    disp_mag = sqrt.(sum(displacement.^2, dims=2))[:,1]  # length nNodes

    # Auto-adjust scale factor if needed
    max_disp = maximum(abs.(displacement))
    if max_disp > 0
        max_dim = maximum([
            maximum(nodes[:,1]) - minimum(nodes[:,1]),
            maximum(nodes[:,2]) - minimum(nodes[:,2]),
            maximum(nodes[:,3]) - minimum(nodes[:,3])
        ])
        if scale * max_disp > max_dim * 5
            @warn "Scale factor causes very large deformation => auto reducing."
            scale = 0.5f0 * max_dim / max_disp
        end
    end

    # Deformed coordinates
    deformed_nodes = copy(nodes)
    @inbounds for i in 1:nNodes
        deformed_nodes[i,1] += scale*displacement[i,1]
        deformed_nodes[i,2] += scale*displacement[i,2]
        deformed_nodes[i,3] += scale*displacement[i,3]
    end
    
    # Check deformed nodes for extreme values
    deformed_nodes = sanitize_data(deformed_nodes)

    # Ensure principal_field and vonmises_field are properly sized
    if size(principal_field, 2) < nElem
        principal_field = hcat(principal_field, zeros(Float32, 3, nElem - size(principal_field, 2)))
    end
    
    vonmises_field = ensure_array_size(vonmises_field, nElem)
    
    if size(full_stress_voigt, 2) < nElem
        full_stress_voigt = hcat(full_stress_voigt, zeros(Float32, 6, nElem - size(full_stress_voigt, 2)))
    end
    
    # Filter stress data to valid elements
    principal_field_valid = principal_field[:, valid_elements]
    vonmises_field_valid = vonmises_field[valid_elements]
    full_stress_voigt_valid = full_stress_voigt[:, valid_elements]

    # Base file name
    if endswith(lowercase(filename), ".vtk") || endswith(lowercase(filename), ".vtu")
        base_filename = filename[1:end-4]
    else
        base_filename = filename
    end
    
    orig_filename = base_filename * "_original.vtk"
    deform_filename = base_filename * "_deformed.vtk"

    # ----------------------------
    # EXPORT ORIGINAL MESH
    # ----------------------------
    try
        open(orig_filename, "w") do file
            # Write VTK header
            write(file, "# vtk DataFile Version 3.0\n")
            write(file, "HEXA FEM Solution (Original Mesh)\n")
            write(file, "ASCII\n")
            write(file, "DATASET UNSTRUCTURED_GRID\n")
            
            # Write node coordinates
            write(file, "POINTS $(nNodes) float\n")
            for i in 1:nNodes
                @printf(file, "%.6f %.6f %.6f\n", nodes[i, 1], nodes[i, 2], nodes[i, 3])
            end
            
            # Write cell connectivity
            write(file, "\nCELLS $(nElem_valid) $(nElem_valid * 9)\n")
            for idx in valid_elements
                @printf(file, "8 %d %d %d %d %d %d %d %d\n", 
                       elements[idx, 1]-1, elements[idx, 2]-1, elements[idx, 3]-1, elements[idx, 4]-1,
                       elements[idx, 5]-1, elements[idx, 6]-1, elements[idx, 7]-1, elements[idx, 8]-1)
            end
            
            # Write cell types
            write(file, "\nCELL_TYPES $(nElem_valid)\n")
            for _ in 1:nElem_valid
                write(file, "12\n")  # VTK_HEXAHEDRON
            end
            
            # Write point data
            write(file, "\nPOINT_DATA $(nNodes)\n")
            
            # Displacement vector
            write(file, "VECTORS Displacement float\n")
            for i in 1:nNodes
                @printf(file, "%.6e %.6e %.6e\n", displacement[i, 1], displacement[i, 2], displacement[i, 3])
            end
            
            # Displacement magnitude
            write(file, "\nSCALARS Displacement_Magnitude float 1\n")
            write(file, "LOOKUP_TABLE default\n")
            for i in 1:nNodes
                @printf(file, "%.6e\n", disp_mag[i])
            end
            
            # Forces
            write(file, "\nVECTORS Force float\n")
            for i in 1:nNodes
                @printf(file, "%.6e %.6e %.6e\n", forces[i, 1], forces[i, 2], forces[i, 3])
            end
            
            # BC data if available
            if size(bc_indicator, 1) == nNodes
                ncols_bc = min(size(bc_indicator, 2), 3)
                if ncols_bc >= 2
                    write(file, "\nVECTORS BC float\n")
                    for i in 1:nNodes
                        if ncols_bc == 2
                            @printf(file, "%.6e %.6e 0.0\n", bc_indicator[i, 1], bc_indicator[i, 2])
                        else
                            @printf(file, "%.6e %.6e %.6e\n", 
                                   bc_indicator[i, 1], bc_indicator[i, 2], bc_indicator[i, 3])
                        end
                    end
                else
                    write(file, "\nSCALARS BC float 1\n")
                    write(file, "LOOKUP_TABLE default\n")
                    for i in 1:nNodes
                        @printf(file, "%.6e\n", bc_indicator[i, 1])
                    end
                end
            end
            
            # Write cell data
            write(file, "\nCELL_DATA $(nElem_valid)\n")
            
            # Von Mises stress
            write(file, "SCALARS Von_Mises_Stress float 1\n")
            write(file, "LOOKUP_TABLE default\n")
            for i in 1:nElem_valid
                @printf(file, "%.6e\n", vonmises_field_valid[i])
            end
            
            # Principal stresses as a vector
            write(file, "\nVECTORS Principal_Stress float\n")
            for i in 1:nElem_valid
                @printf(file, "%.6e %.6e %.6e\n", 
                       principal_field_valid[1, i], principal_field_valid[2, i], principal_field_valid[3, i])
            end
            
            # Full stress tensor components as separate scalars
            stress_names = ["Stress_XX", "Stress_YY", "Stress_ZZ", "Stress_XY", "Stress_YZ", "Stress_XZ"]
            for idx in 1:6
                write(file, "\nSCALARS $(stress_names[idx]) float 1\n")
                write(file, "LOOKUP_TABLE default\n")
                for i in 1:nElem_valid
                    @printf(file, "%.6e\n", full_stress_voigt_valid[idx, i])
                end
            end
        end
        
        println("Successfully exported original solution to $orig_filename (ASCII format)")
    catch e
        @error "Failed to save original mesh VTK file: $e"
        println("Error details: ", e)
    end

    # ----------------------------
    # EXPORT DEFORMED MESH
    # ----------------------------
    try
        open(deform_filename, "w") do file
            # Write VTK header
            write(file, "# vtk DataFile Version 3.0\n")
            write(file, "HEXA FEM Solution (Deformed Mesh, scale=$(scale))\n")
            write(file, "ASCII\n")
            write(file, "DATASET UNSTRUCTURED_GRID\n")
            
            # Write deformed node coordinates
            write(file, "POINTS $(nNodes) float\n")
            for i in 1:nNodes
                @printf(file, "%.6f %.6f %.6f\n", 
                       deformed_nodes[i, 1], deformed_nodes[i, 2], deformed_nodes[i, 3])
            end
            
            # Write cell connectivity
            write(file, "\nCELLS $(nElem_valid) $(nElem_valid * 9)\n")
            for idx in valid_elements
                @printf(file, "8 %d %d %d %d %d %d %d %d\n", 
                       elements[idx, 1]-1, elements[idx, 2]-1, elements[idx, 3]-1, elements[idx, 4]-1,
                       elements[idx, 5]-1, elements[idx, 6]-1, elements[idx, 7]-1, elements[idx, 8]-1)
            end
            
            # Write cell types
            write(file, "\nCELL_TYPES $(nElem_valid)\n")
            for _ in 1:nElem_valid
                write(file, "12\n")  # VTK_HEXAHEDRON
            end
            
            # Write point data
            write(file, "\nPOINT_DATA $(nNodes)\n")
            
            # Displacement vector
            write(file, "VECTORS Displacement float\n")
            for i in 1:nNodes
                @printf(file, "%.6e %.6e %.6e\n", displacement[i, 1], displacement[i, 2], displacement[i, 3])
            end
            
            # Displacement magnitude
            write(file, "\nSCALARS Displacement_Magnitude float 1\n")
            write(file, "LOOKUP_TABLE default\n")
            for i in 1:nNodes
                @printf(file, "%.6e\n", disp_mag[i])
            end
            
            # Write cell data
            write(file, "\nCELL_DATA $(nElem_valid)\n")
            
            # Von Mises stress
            write(file, "SCALARS Von_Mises_Stress float 1\n")
            write(file, "LOOKUP_TABLE default\n")
            for i in 1:nElem_valid
                @printf(file, "%.6e\n", vonmises_field_valid[i])
            end
            
            # Principal stresses as a vector
            write(file, "\nVECTORS Principal_Stress float\n")
            for i in 1:nElem_valid
                @printf(file, "%.6e %.6e %.6e\n", 
                       principal_field_valid[1, i], principal_field_valid[2, i], principal_field_valid[3, i])
            end
        end
        
        println("Successfully exported deformed solution to $deform_filename (ASCII format)")
    catch e
        @error "Failed to save deformed mesh VTK file: $e"
        println("Error details: ", e)
    end

    return nothing
end

end # module ExportVTK