import cupy as cp
import trimesh

class MarchingTetrahedra:
    """ GPU-Accelerated Marching Tetrahedra Algorithm using CuPy """

    TETRAHEDRA = cp.array([
        [0, 5, 1, 6], [0, 5, 6, 4], [0, 6, 2, 4],
        [6, 2, 4, 3], [5, 6, 4, 7], [2, 6, 3, 7]
    ], dtype=cp.int32)

    EDGE_VERTICES = cp.array([
        [0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]
    ], dtype=cp.int32)

    ISO_VALUE = 0.0

    # Kernel for GPU-based interpolation
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

    def __init__(self, points, resolution=64, padding=0.1):
        """
        Initialize the Marching Tetrahedra algorithm.
        :param points: (cp.ndarray) Nx3 array of 3D points.
        :param resolution: (int) Grid resolution along each axis.
        :param padding: (float) Padding around the bounding box.
        """
        self.points = points
        self.resolution = resolution
        self.padding = padding
        self.grid = None
        self.values = None

    def generate_grid(self):
        """ Generate a 3D grid from the bounding box of the input points. """
        min_point = cp.min(self.points, axis=0) - self.padding
        max_point = cp.max(self.points, axis=0) + self.padding

        x = cp.linspace(min_point[0], max_point[0], self.resolution)
        y = cp.linspace(min_point[1], max_point[1], self.resolution)
        z = cp.linspace(min_point[2], max_point[2], self.resolution)
        X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

        self.grid = cp.stack([X, Y, Z], axis=-1)

    def apply_scalar_function(self, scalar_function):
        """
        Apply a user-defined scalar function to compute grid values.
        :param scalar_function: Function(x, y, z) -> scalar field value
        """
        X, Y, Z = self.grid[..., 0], self.grid[..., 1], self.grid[..., 2]
        self.values = scalar_function(X, Y, Z)

    def extract_surface(self):
        """ Run Marching Tetrahedra and extract the isosurface mesh. """
        nx, ny, nz = self.grid.shape[:3]

        x, y, z = cp.meshgrid(cp.arange(nx - 1), cp.arange(ny - 1), cp.arange(nz - 1), indexing='ij')
        x, y, z = x.ravel(), y.ravel(), z.ravel()

        cube_offsets = cp.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=cp.int32)

        cube_vertices = self.grid[x[:, None] + cube_offsets[:, 0],
                                  y[:, None] + cube_offsets[:, 1],
                                  z[:, None] + cube_offsets[:, 2]]

        cube_values = self.values[x[:, None] + cube_offsets[:, 0],
                                  y[:, None] + cube_offsets[:, 1],
                                  z[:, None] + cube_offsets[:, 2]]

        triangles = self._process_tetrahedra(cube_vertices, cube_values)
        return triangles

    def _process_tetrahedra(self, cube_vertices, cube_values):
        """ Internal method to process tetrahedra for isosurface extraction. """
        tetra_vertices = cube_vertices[:, self.TETRAHEDRA]
        tetra_values = cube_values[:, self.TETRAHEDRA]

        inside = tetra_values >= self.ISO_VALUE
        num_inside = cp.sum(inside, axis=2)

        valid_mask = (num_inside > 0) & (num_inside < 4)
        if not valid_mask.any():
            return None  

        tetra_vertices = tetra_vertices[valid_mask]
        tetra_values = tetra_values[valid_mask]

        edge_mask = inside[:, :, self.EDGE_VERTICES[:, 0]] != inside[:, :, self.EDGE_VERTICES[:, 1]]
        intersecting_edges = self.EDGE_VERTICES[edge_mask]

        v1 = tetra_vertices[:, :, intersecting_edges[:, 0]]
        v2 = tetra_vertices[:, :, intersecting_edges[:, 1]]
        val1 = tetra_values[:, :, intersecting_edges[:, 0]]
        val2 = tetra_values[:, :, intersecting_edges[:, 1]]

        interp_x, interp_y, interp_z = self.interp_kernel(
            v1[:, :, 0], v1[:, :, 1], v1[:, :, 2],
            v2[:, :, 0], v2[:, :, 1], v2[:, :, 2],
            val1, val2
        )

        return cp.stack([interp_x, interp_y, interp_z], axis=-1)

    @staticmethod
    def save_mesh(vertices, filename="output.obj"):
        """ Save extracted mesh to an OBJ file. """
        if vertices is not None:
            vertices_np = vertices.get()
            mesh = trimesh.Trimesh(vertices=vertices_np.reshape(-1, 3), process=False)
            mesh.export(filename)
            print(f"Mesh saved as '{filename}'")
