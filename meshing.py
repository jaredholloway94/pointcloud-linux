import networkx as nx
import cupy as cp
from cupyx.scipy.spatial.distance import cdist
# import trimesh

points = cp.genfromtxt("bunny.csv", delimiter=",", dtype=cp.float32)
points = cp.unique(points,axis=0)


def approx_surface(points,K=5):

    N = points.shape[0]

    point_distances = cdist(points,points)
    point_neighbors = point_distances.argsort(axis=1)[:, 1:K+1]

    tp_origins = cp.mean(points[point_neighbors], axis=1)

    tp_distances = cdist(tp_origins, tp_origins)
    tp_neighbors = tp_distances.argsort(axis=1)[:, 1:K+1]
    tp_adjacency = cp.zeros((N,N), dtype=cp.float32)
    tp_adjacency[cp.arange(N).repeat(K), tp_neighbors.ravel()] = 1

    tp_deviations = points[point_neighbors] - tp_origins[:, None, :]
    tp_outer_products = tp_deviations[..., :, None] * tp_deviations[..., None, :]
    tp_cv_matrices = cp.sum(tp_outer_products, axis=1)
    tp_eigvals, tp_eigvecs = cp.linalg.eigh(tp_cv_matrices)
    tp_normals = tp_eigvecs[:, :, 0]
    tp_weights = 1 - cp.abs(cp.dot(tp_normals, tp_normals.T))

    tp_weighted_adj = tp_adjacency * tp_weights

    tp_riem_graph = nx.from_numpy_array(tp_weighted_adj)
    tp_riem_mst = nx.minimum_spanning_tree(tp_riem_graph)
    tp_riem_graph_root_node = int(cp.argmax(tp_origins[:, 2]))
    tp_riem_mst_dfs = nx.dfs_edges(tp_riem_mst,tp_riem_graph_root_node)

    # flip inverted tangent planes to get consistent orientation
    tp_riem_mst_dfs = cp.array(list(tp_riem_mst_dfs))
    a, b = tp_riem_mst_dfs.T
    neg_mask = (cp.sum(tp_normals[a] * tp_normals[b], axis=1) < 0)
    tp_normals[b[neg_mask]] *= -1

    M = tp_origins, tp_normals

    return M


def signed_dist(p, M, rho_delta=0.06):

    # Find index of the closest tangent plane
    i = cp.argmin(cp.linalg.norm(M[0] - p, axis=1))  # Scalar index

    # Retrieve the closest tangent plane's origin and normal
    o_i = M[0][i]  # Shape: (3,)
    n_i = M[1][i]  # Shape: (3,)

    # Compute projection of p onto the tangent plane
    z_i = o_i - cp.dot((p - o_i), n_i) * n_i

    obj_size = cp.linalg.norm(cp.max(M[0],axis=0) - cp.min(M[0],axis=0))
    expected_point_dist = rho_delta * obj_size

    if cp.linalg.norm(z_i - o_i) > expected_point_dist:
        signed_dist = cp.inf
    else:
        signed_dist = cp.dot((p - o_i), n_i)

    return signed_dist





# from marching_tetrahedra import MarchingTetrahedra

# mt = MarchingTetrahedra(points)
# mt.generate_grid()
# mt.apply_scalar_function(signed_dist_func)
# mesh_vertices = mt.extract_surface()
# mt.save_mesh(mesh_vertices, "marching_tetrahedra_output.obj")