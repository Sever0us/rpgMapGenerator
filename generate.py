import numpy as np 
import random
from math import inf
from scipy.spatial import Voronoi, Delaunay
from matplotlib import pyplot as plt

def centroid(verts):
    xs, ys = verts[:, 0], verts[:, 1]
    sxs, sys = np.roll(xs, 1), np.roll(ys, 1)
    A = 0.5 * np.sum(np.multiply(xs, sys) - np.multiply(sxs, ys))
    cx = (1/(6*A)) * np.sum( 
        np.multiply(
            (xs + sxs),
            (np.multiply(xs, sys) - np.multiply(sxs, ys))
        )
    )
    cy = (1/(6*A)) * np.sum( 
        np.multiply(
            (ys + sys),
            (np.multiply(xs, sys) - np.multiply(sxs, ys))
        )
    )
    return cx, cy

def relaxation(points):
    voronoi = Voronoi(
        points=points
    )
    relaxed_points = []

    for i, pr in enumerate(voronoi.point_region):
        point, reigion = voronoi.points[i], voronoi.regions[pr]
        
        # Ignore points not associated with a finite region
        if pr == -1 or -1 in reigion:
            relaxed_points.append(point)
            continue

        # Calculate and substitute appropriate centroid
        region_verts = voronoi.vertices[reigion]
        cx, cy = centroid(region_verts)
        if 0 < cx < 1 and 0 < cy < 1:
            relaxed_points.append((cx, cy))
        else:
            relaxed_points.append((point))

    return np.asarray(relaxed_points)

def construct_lookups(points, tris):
    indices, indptr = tris.vertex_neighbor_vertices

    def find_index(point):
        return np.where((points == (point[0], point[1])).all(axis=1))[0][0]

    def find_connected(point):
        index = find_index(point)
        return points[indptr[indices[index]:indices[index+1]]]

    return find_connected, find_index

def generate_mesh(n, iterations=1):
    print('Generating random mesh')
    # Generate random points
    points = np.random.random([n, 2])  

    # Relax points
    print('Relaxing mesh points')
    for _ in range(iterations):
        points = relaxation(points)

    # Get triangles
    print('Constructing lookups')
    triangles = Delaunay(points)
    find_connected, find_index = construct_lookups(points, triangles)

    return points, find_connected, find_index


if __name__ == "__main__":
    points, find_neighbours, find_index  = generate_mesh(15, 5)
