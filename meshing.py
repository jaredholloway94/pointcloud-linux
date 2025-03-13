import cupy as cp
import cupyx.scipy as sp
import cupyx.scipy.spatial
import networkx as nx

points = cp.genfromtxt("bunny.csv", delimiter=",", dtype=cp.float32)
points = cp.unique(points,axis=0)

K = 5

point_distances = sp.spatial.distance.cdist(points,points)
point_neighbors = point_distances.argsort(axis=1)[:, 1:K+1]
point_neighbor_coordinates = points[point_neighbors]

tangent_plane_origins = cp.mean(point_neighbor_coordinates, axis=1)
tpo_deviations = point_neighbor_coordinates - tangent_plane_origins[:, None, :]
tpo_outer_products = tpo_deviations[..., :, None] * tpo_deviations[..., None, :]
tpo_cv_matrices = cp.sum(tpo_outer_products, axis=1)
tpo_eigvals, tpo_eigvecs = cp.linalg.eigh(tpo_cv_matrices)
tpo_normals = tpo_eigvecs[:, :, 0]

tpo_distances = sp.spatial.distance.cdist(tangent_plane_origins, tangent_plane_origins)
tpo_neighbors = tpo_distances.argsort(axis=1)[:, 1:K+1]
tpo_neighbor_coordinates = tangent_plane_origins[tpo_neighbors]


def tp_graph_edge_weight(i,j):
    return ( 1 - cp.abs(cp.dot(tpo_normals[int(i)],tpo_normals[int(j)])) )


tp_graph = nx.Graph()
tp_graph.add_nodes_from([
    (i, {'x':t[0], 'y':t[1], 'z':t[2]})
    for i,t in enumerate(tangent_plane_origins)
    ]
)
tp_graph.add_edges_from([
    ( i, int(j), {'weight': tp_graph_edge_weight(i,j)} )
    for i,t in enumerate(tpo_neighbors)
    for j in t
    ]
)

tp_graph_root_node = tangent_plane_origins[cp.argmax(tangent_plane_origins[:, 2])]

tp_graph_mst = nx.minimum_spanning_tree(tp_graph)