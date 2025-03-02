module HEXA

using LinearAlgebra, SparseArrays, Printf, Base.Threads, JSON, Dates
using CUDA  # For optional GPU checks

include("Helpers.jl")
include("Element.jl")
include("Mesh.jl")
include("Boundary.jl")
include("ExportVTK.jl")
include("Stress.jl")
include("CPUSolver.jl")
include("GPUSolver.jl")
include("IterativeSolver.jl")
include("DirectSolver.jl")

using .Helpers
using .Element
using .Mesh
using .Boundary
using .ExportVTK
using .Stress
using .CPUSolver
using .GPUSolver
using .IterativeSolver: solve_system_iterative
using .DirectSolver: solve_system as solve_system_direct

function __init__()
    println("HEXA Finite Element Solver initialized")
    println("Active Threads = $(Threads.nthreads())")
    println("Clearing GPU memory...")
    Helpers.clear_gpu_memory()
end

"""
    load_configuration(filename::String)

Load and parse a JSON configuration file.
"""
function load_configuration(filename::String)
    if !isfile(filename)
        error("Configuration file '$(filename)' not found")
    end
    open(filename, "r") do io
        config_str = read(io, String)
        return JSON.parse(config_str)
    end
end

"""
    setup_geometry(config)

Process the geometry configuration and return parameters for mesh generation.
Now the geometry section may contain multiple shape definitions (each with a "type" field).
"""
function setup_geometry(config)
    # Extract basic dimensions
    length_x = config["geometry"]["length_x"]
    length_y = config["geometry"]["length_y"]
    length_z = config["geometry"]["length_z"]
    target_elem_count = config["geometry"]["target_elem_count"]
    
    println("Domain dimensions:")
    println("  X: 0 to $(length_x)")
    println("  Y: 0 to $(length_y)")
    println("  Z: 0 to $(length_z)")
    
    # Process shape definitions: any key that is not one of the basic geometry parameters
    shapes_add = Any[]
    shapes_remove = Any[]
    for (key, shape) in config["geometry"]
        if key in ["length_x", "length_y", "length_z", "target_elem_count"]
            continue
        end
        if haskey(shape, "type")
            action = lowercase(get(shape, "action", "remove"))
            if action == "add"
                push!(shapes_add, shape)
            elseif action == "remove"
                push!(shapes_remove, shape)
            else
                @warn "Unknown action for shape '$key'. Defaulting to 'remove'."
                push!(shapes_remove, shape)
            end
        else
            @warn "Geometry key '$key' does not have a 'type' field; skipping."
        end
    end

    println("Found $(length(shapes_add)) shapes to add and $(length(shapes_remove)) shapes to remove.")
    
    # Calculate element distribution
    nElem_x, nElem_y, nElem_z, dx, dy, dz, actual_elem_count =
        calculate_element_distribution(length_x, length_y, length_z, target_elem_count)
    
    println("Mesh parameters:")
    println("  Domain: $(length_x) x $(length_y) x $(length_z) meters")
    println("  Elements: $(nElem_x) x $(nElem_y) x $(nElem_z) = $(actual_elem_count)")
    println("  Element sizes: $(dx) x $(dy) x $(dz)")
    
    return (
        nElem_x = nElem_x, 
        nElem_y = nElem_y, 
        nElem_z = nElem_z,
        dx = dx,
        dy = dy,
        dz = dz,
        shapes_to_add = shapes_add,
        shapes_to_remove = shapes_remove,
        actual_elem_count = actual_elem_count
    )
end

"""
    choose_solver(nNodes, nElem)

Determine the appropriate solver type based on problem size and available hardware.
If a GPU is available with sufficient memory (using Helpers.has_enough_gpu_memory),
returns `:gpu`. Otherwise, if the number of elements is small, returns `:direct`;
else returns `:matrix_free`.
"""
function choose_solver(nNodes, nElem)
    if CUDA.functional() && Helpers.has_enough_gpu_memory(nNodes, nElem)
        println("GPU is available with sufficient memory. Using GPU solver.")
        return :gpu
    else
        if CUDA.functional()
            println("GPU is available but not enough memory; falling back to CPU solver.")
        else
            println("No GPU available; using CPU solver.")
        end
        
        if nElem < 100_001
            println("Using Direct Solver (since nElem = $nElem).")
            return :direct
        else
            println("Using Matrix-Free Iterative Solver (since nElem = $nElem).")
            return :matrix_free
        end
    end
end

"""
    run_main(config_file=nothing)

Run the main HEXA simulation using the configuration from the specified JSON file.
If no file is provided, it uses "config.json" from the current directory.
"""
function run_main(config_file=nothing)
    if config_file === nothing
        config_file = "config.json"
    end
    
    println("Loading configuration from: $config_file")
    config = load_configuration(config_file)
    
    geom = setup_geometry(config)
    
    println("\nGenerating mesh with geometric shape processing...")
    nodes, elements, dims = generate_mesh(
        geom.nElem_x, geom.nElem_y, geom.nElem_z;
        dx = geom.dx, dy = geom.dy, dz = geom.dz,
        shapes_to_add = geom.shapes_to_add,
        shapes_to_remove = geom.shapes_to_remove,
        remove_unused_nodes = true
    )
    
    nNodes = size(nodes, 1)
    nElem = size(elements, 1)
    println("Final mesh statistics:")
    println("  Nodes: $(nNodes)")
    println("  Elements: $(nElem)")
    
    bc_data = config["boundary_conditions"]
    bc_indicator = get_bc_indicator(nNodes, nodes, bc_data)
    
    E = Float32(config["material"]["E"])
    nu = Float32(config["material"]["nu"])
    println("Material properties: E=$(E), nu=$(nu)")
    
    ndof = nNodes * 3
    F = zeros(Float32, ndof)
    forces_data = config["external_forces"]
    apply_external_forces!(F, forces_data, nodes, elements)
    
    density = ones(Float32, nElem)
    
    solver = choose_solver(nNodes, nElem)
    
    U_full = if solver == :direct
        solve_system_direct(nodes, elements, E, nu, bc_indicator, F; density=density)
    elseif solver == :gpu
        solve_system_iterative(nodes, elements, E, nu, bc_indicator, F;
                             solver_type=:gpu, max_iter=15000, tol=1e-3,
                             density=density)
    else
        solve_system_iterative(nodes, elements, E, nu, bc_indicator, F;
                             solver_type=:matrix_free, max_iter=500, tol=1e-3,
                             density=density)
    end
    println("Solution computed.")
    
    principal_field, vonmises_field, full_stress_voigt =
        compute_stress_field(nodes, elements, U_full, E, nu)
    
    base_name = splitext(basename(config_file))[1]
    mesh_filename = "$(base_name)_mesh.vtu"
    solution_filename = "$(base_name)_solution.vtu"
    
    export_mesh(nodes, elements;
              bc_indicator=bc_indicator,
              filename=mesh_filename)
    println("Exported mesh + BC to $mesh_filename")
    
    export_solution(nodes, elements, U_full, F, bc_indicator,
                  principal_field, vonmises_field, full_stress_voigt;
                  scale=1.0f0,
                  filename=solution_filename)
    println("Solution exported to $(splitext(solution_filename)[1])_original.vtu and $(splitext(solution_filename)[1])_deformed.vtu")
    
    println("Clearing GPU memory...")
    Helpers.clear_gpu_memory()
    
    return (
        nodes = nodes, 
        elements = elements, 
        displacements = U_full,
        principal_stress = principal_field,
        vonmises_stress = vonmises_field, 
        stress_tensor = full_stress_voigt
    )
end

end  # module HEXA

using .HEXA
# Change the configuration file path as needed.
HEXA.run_main(raw"F:\PhD\01_DEV\JULIA\04_HEXAS\HEXAS_GitHub_folder\config.json")
