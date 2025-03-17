# from chatgpt

import cupy as cp
import numpy as np
import trimesh

# Define tetrahedra decomposition of a cube (6 per cube)
TETRAHEDRA = cp.array([
    [0, 5, 1, 6], [0, 5, 6, 4], [0, 6, 2, 4],
    [6, 2, 4, 3], [5, 6, 4, 7], [2, 6, 3, 7]
], dtype=cp.int32)

# Define edge indices for tetrahedra
EDGE_VERTICES = cp.array([
    [0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]
], dtype=cp.int32)

# Isovalue
ISO_VALUE = 0.0

# Kernel for interpolating vertices on edges
interp_kernel = cp.ElementwiseKernel(
    'float32 v1_x, float32 v1_y, float32 v1_z, float32 v2_x, float32 v2_y, float32 v2_z, float32 val1, float32 val2',
    'float32 out_x, float32 out_y, float32 out_z',
    '''
    float t = (0.0 - val1) / (val2 - val1 + 1e-6);
    out_x = v1_x + t * (v2_x - v1_x);
    out_y = v1_y + t * (v2_y - v1_y);
    out_z = v1_z + t * (v2_z - v1_z);
    ''',
    'interpolate_vertices'
)


def generate_grid(size=64, bounds=(-1, 1)):
    """ Generate a 3D grid and a scalar field. """
    x = cp.linspace(bounds[0], bounds[1], size)
    y = cp.linspace(bounds[0], bounds[1], size)
    z = cp.linspace(bounds[0], bounds[1], size)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

    # Example scalar field (sphere function)
    values = X**2 + Y**2 + Z**2 - 0.5**2
    grid = cp.stack([X, Y, Z], axis=-1)

    return grid, values


def process_tetrahedra(cube_vertices, cube_values):
    """ Extract isosurface triangles using vectorized Marching Tetrahedra. """
    tetra_vertices = cube_vertices[:, TETRAHEDRA]  # Shape: (num_cubes, 6, 4, 3)
    tetra_values = cube_values[:, TETRAHEDRA]  # Shape: (num_cubes, 6, 4)

    # Create masks for inside/outside classification
    inside = tetra_values >= ISO_VALUE
    num_inside = cp.sum(inside, axis=2)  # Count inside vertices per tetrahedron

    # Only process tetrahedra that have a partial intersection with the isosurface
    valid_mask = (num_inside > 0) & (num_inside < 4)

    if not valid_mask.any():
        return None  # No triangles extracted

    tetra_vertices = tetra_vertices[valid_mask]
    tetra_values = tetra_values[valid_mask]

    # Get edge indices where the surface intersects
    edge_mask = inside[:, :, EDGE_VERTICES[:, 0]] != inside[:, :, EDGE_VERTICES[:, 1]]
    intersecting_edges = EDGE_VERTICES[edge_mask]

    # Extract vertex pairs for interpolation
    v1 = tetra_vertices[:, :, intersecting_edges[:, 0]]
    v2 = tetra_vertices[:, :, intersecting_edges[:, 1]]
    val1 = tetra_values[:, :, intersecting_edges[:, 0]]
    val2 = tetra_values[:, :, intersecting_edges[:, 1]]

    # Compute interpolated vertices on the GPU
    interp_x, interp_y, interp_z = interp_kernel(
        v1[:, :, 0], v1[:, :, 1], v1[:, :, 2],
        v2[:, :, 0], v2[:, :, 1], v2[:, :, 2],
        val1, val2
    )

    # Concatenate interpolated vertices
    vertices = cp.stack([interp_x, interp_y, interp_z], axis=-1)

    return vertices


def marching_tetrahedra(grid, values):
    """ Vectorized GPU implementation of Marching Tetrahedra. """
    nx, ny, nz = grid.shape[:3]

    # Generate cube indices for the entire grid (excluding last layer)
    x, y, z = cp.meshgrid(cp.arange(nx - 1), cp.arange(ny - 1), cp.arange(nz - 1), indexing='ij')
    x, y, z = x.ravel(), y.ravel(), z.ravel()

    # Define cube corner offsets
    cube_offsets = cp.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=cp.int32)

    # Compute absolute cube vertex positions and values
    cube_vertices = grid[x[:, None] + cube_offsets[:, 0],
                         y[:, None] + cube_offsets[:, 1],
                         z[:, None] + cube_offsets[:, 2]]

    cube_values = values[x[:, None] + cube_offsets[:, 0],
                         y[:, None] + cube_offsets[:, 1],
                         z[:, None] + cube_offsets[:, 2]]

    # Process tetrahedra in parallel
    triangles = process_tetrahedra(cube_vertices, cube_values)

    return triangles


# Run the optimized algorithm
grid, values = generate_grid(size=64)
triangles = marching_tetrahedra(grid, values)

# Convert to NumPy for visualization
triangles_np = triangles.get() if triangles is not None else np.array([])

# Save mesh using trimesh
if triangles_np.size > 0:
    mesh = trimesh.Trimesh(vertices=triangles_np.reshape(-1, 3), process=False)
    mesh.export('marching_tetrahedra_output.obj')
    print("Mesh saved as 'marching_tetrahedra_output.obj'")
