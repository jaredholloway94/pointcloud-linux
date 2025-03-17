import networkx as nx
import cupy as cp
from cupyx.scipy.spatial.distance import cdist
from copy import deepcopy

points = cp.genfromtxt("bunny.csv", delimiter=",", dtype=cp.float32)
points = cp.unique(points,axis=0)

N = points.shape[0]
K = 5

point_distances = cdist(points,points)
point_neighbors = point_distances.argsort(axis=1)[:, 1:K+1]
point_neighbor_coordinates = points[point_neighbors]

tp_origins = cp.mean(point_neighbor_coordinates, axis=1)

tp_distances = cdist(tp_origins, tp_origins)
tp_neighbors = tp_distances.argsort(axis=1)[:, 1:K+1]
tp_adjacency = cp.zeros((N,N), dtype=cp.float32)
tp_adjacency[cp.arange(N).repeat(K), tp_neighbors.ravel()] = 1

tp_deviations = point_neighbor_coordinates - tp_origins[:, None, :]
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

tp_normals_oriented = deepcopy(tp_normals)
for a,b in tp_riem_mst_dfs:
    if cp.dot(tp_normals_oriented[a],tp_normals_oriented[b]) < 0:
        tp_normals_oriented[b] = -1 * tp_normals_oriented[b]





# get pointcloud bounding box
points_bb = cp.stack( (cp.min(points, axis=0),cp.max(points, axis=0)) )
