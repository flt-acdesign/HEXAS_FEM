### Description of the Code

This repository contains a **3D Finite Element Analysis (FEA) solver** implemented in Julia, specifically designed for solving structural mechanics problems using hexahedral (hex) elements. The solver is capable of handling complex geometries, applying boundary conditions, and computing displacements, stresses, and strains. It supports both **CPU and GPU-based solvers**, including direct and iterative methods, and provides functionality for exporting results to VTK files for visualization in tools like ParaView.

The code is modular and organized into several submodules, each responsible for a specific aspect of the FEA pipeline:

1. **Mesh Generation**: Creates structured hexahedral meshes and supports adding/removing geometric shapes (e.g., spheres, boxes) to create complex geometries.
2. **Boundary Conditions**: Handles the application of boundary conditions (e.g., fixed nodes) and external forces.
3. **Element Stiffness**: Computes the stiffness matrix for hexahedral elements using numerical integration.
4. **Solvers**:
   - **Direct Solver**: Solves the system using LU factorization (CPU-only).
   - **Iterative Solver**: Uses Conjugate Gradient (CG) methods, with support for matrix-free and GPU-accelerated solvers.
5. **Stress Computation**: Computes principal stresses, von Mises stress, and full stress tensors for each element.
6. **Export to VTK**: Exports mesh and solution data to VTK files for visualization.

### Key Features

- **Flexible Geometry**: Supports adding and removing geometric shapes (spheres, boxes) to create complex meshes.
- **Boundary Conditions**: Allows specifying boundary conditions by node indices or spatial locations.
- **Material Properties**: Supports isotropic materials with user-defined Young's modulus (E) and Poisson's ratio (nu).
- **Solvers**:
  - **Direct Solver**: Efficient for small to medium-sized problems.
  - **Iterative Solver**: Suitable for large-scale problems, with options for matrix-free and GPU-accelerated solvers.
- **Stress Analysis**: Computes principal stresses, von Mises stress, and full stress tensors.
- **Visualization**: Exports results to VTK files for visualization in ParaView.

### User Instructions

1. **Installation**:
   - Ensure Julia is installed on your system.
   - Install required Julia packages by running:
     ```julia
     using Pkg
     Pkg.add("LinearAlgebra")
     Pkg.add("SparseArrays")
     Pkg.add("CUDA")
     Pkg.add("JSON")
     Pkg.add("Printf")
     Pkg.add("Dates")
     ```

2. **Configuration**:
   - Modify the `config.json` file to define the problem:
     - **Geometry**: Define the domain size, target element count, and any additional shapes (e.g., spheres, boxes).
     - **Boundary Conditions**: Specify fixed nodes or regions.
     - **External Forces**: Define applied forces and their locations.
     - **Material Properties**: Set Young's modulus (E) and Poisson's ratio (nu).

3. **Running the Solver**:
   - Run the solver by executing the `Main.jl` script:
     ```julia
     include("Main.jl")
     HEXA.run_main("path/to/config.json")
     ```
   - The solver will automatically choose the appropriate solver (direct, iterative, or GPU-based) based on the problem size and available hardware.

4. **Output**:
   - The solver generates two VTK files:
     - `mesh_output.vtu`: Contains the mesh and boundary conditions.
     - `solution_output.vtu`: Contains the displacement, stress, and strain fields.
   - Open these files in ParaView to visualize the results.

5. **Customization**:
   - To modify the solver behavior (e.g., switch between direct and iterative solvers), edit the `choose_solver` function in `Main.jl`.
   - To add new geometric shapes or boundary conditions, modify the relevant modules (`Mesh.jl`, `Boundary.jl`).

### Example `config.json`

```json
{
  "geometry": {
    "length_x": 40.0,
    "length_y": 10.0,
    "length_z": 2.0,
    "target_elem_count": 85000,
    "sphere1": {
      "type": "sphere",
      "center": [7.0, 5.0, 4.0],
      "diameter": 8.0,
      "action": "add"
    },
    "sphere2": {
      "type": "sphere",
      "center": [9.0, 5.0, 2.0],
      "diameter": 8.0,
      "action": "remove"
    }
  },
  "boundary_conditions": [
    {
      "location": [0.0, ":", ":"],
      "DoFs": [1, 2, 3]
    }
  ],
  "external_forces": [
    {
      "location": [1.0, 0.0, 0.0],
      "F": [0.0, 10.0, 0.0]
    }
  ],
  "material": {
    "E": 2.1e11,
    "nu": 0.3
  }
}
```

### Notes

- **GPU Support**: The GPU solver requires a CUDA-capable GPU and the `CUDA.jl` package. Ensure your system meets these requirements if you plan to use GPU acceleration.
- **Performance**: For large problems, the iterative solver (especially the GPU-accelerated one) is recommended for better performance.
- **Visualization**: Use ParaView or any other VTK-compatible tool to visualize the results.

This solver is designed for educational and research purposes and can be extended to support more advanced features like nonlinear materials, dynamic analysis, or multi-physics simulations.
