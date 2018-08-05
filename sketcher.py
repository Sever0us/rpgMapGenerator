from heightmap import HeightMap
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d
from functools import reduce
import random 

class Sketcher:
    def __init__(self, heightmap):
        self.heightmap = heightmap

    def setup_sketch(self):
        self.fig = plt.figure(frameon=False)
        self.fig.set_size_inches(5,5)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_xlim([0.1, 0.9])
        self.ax.set_ylim([0.1, 0.9])
        self.ax.set_axis_off()
        self.fig.add_axes(self.ax)

    def get_coastal_points(self):
        # TODO: Seperate points into individual landmasses
        coastal_points = []
        for index, point in enumerate(self.heightmap.mesh):
            if self.heightmap.heights[index] > self.heightmap.sea_level:
                neighbour_index = [
                    self.heightmap.find_index(n)  for n in self.heightmap.neighbours(point)
                ]
                for ni in neighbour_index:
                    if self.heightmap.heights[ni] <= self.heightmap.sea_level:
                        coastal_points.append(point)
                        break
        return np.asarray(coastal_points)

    def seperate_landmasses(self, coastal_points):
        depleted = []
        live = [i for i in range(len(coastal_points))]
        landmasses = []
        while live:
            # Select random coastal point
            rand_index = random.choice(live)
            rand_point = coastal_points[rand_index]
            
            # Get neighbours and cehck for landmass links
            neighbours = self.heightmap.neighbours(rand_point)
            landmass_links = set()
            for landmass_index, landmass in enumerate(landmasses):
                for neighbour in neighbours:
                    if any((neighbour == x).all() for x in landmass):
                        landmass_links.add(landmass_index)
            landmass_links = list(landmass_links)

            # No links
            if len(landmass_links) == 0:
                landmasses.append([rand_point])

            # One link
            elif len(landmass_links) == 1:
                landmasses[landmass_links[0]].append(rand_point)

            # Multiple links
            else:
                unified = [rand_point]

                for landmass_index in landmass_links:
                    unified += landmasses[landmass_index]

                for landmass_index in sorted(landmass_links, reverse=True):
                    del landmasses[landmass_index]
                landmasses.append(unified)

            # Mark point as depleted
            depleted.append(rand_index)
            del live[live.index(rand_index)] 

        return [np.asarray(landmass) for landmass in landmasses]

    def calculate_sea_boundary(self, coastal_loop):
        '''Finds the sea level polygon for a single coastal loop'''
        collection_points = [point for point in coastal_loop]
        collection_heights = [
            self.heightmap.heights[self.heightmap.find_index(point)] 
            for point in coastal_loop
        ]

        # Collect a set of all costal points and first degree neighbours
        for point in coastal_loop:
            for npoint in self.heightmap.neighbours(point):
                ni = self.heightmap.find_index(npoint)
                nheight = self.heightmap.heights[ni]
                if nheight < self.heightmap.sea_level or True:
                    collection_points.append(npoint)
                    collection_heights.append(nheight)

        # Triangulate points
        collection_points = np.asarray(collection_points)
        collection_heights = np.asarray(collection_heights)
        delaunay = Delaunay(collection_points)

        # Find intersection with water plane
        edges = []
        for simplex in delaunay.simplices:
            simplex_points = collection_points[simplex]
            simplex_heights = collection_heights[simplex]

            # Ensure triangle does intersect water
            lower_check = np.less(simplex_heights, self.heightmap.sea_level)
            upper_check = np.greater(simplex_heights, self.heightmap.sea_level)
            if len(np.nonzero(lower_check)[0]) == 2:
                i1, i2 = np.nonzero(lower_check)[0]
                i3 = np.nonzero(upper_check)[0]
            elif len(np.nonzero(upper_check)[0]) == 2:
                i1, i2 = np.nonzero(upper_check)[0]
                i3 = np.nonzero(lower_check)[0]
            else:
                continue

            p1, p2, p3 = simplex_points[i1].flatten(), simplex_points[i2].flatten(), simplex_points[i3].flatten()
            h1, h2, h3 = simplex_heights[i1].flatten(), simplex_heights[i2].flatten(), simplex_heights[i3].flatten()
            
            def get_coefficient(z1, z2):
                return (self.heightmap.sea_level - z2) / (z1 - z2)

            def get_argument(d1, d2, t):
                return (d1 - d2)*t + d2

            t1 = get_coefficient(h1, h3)
            intersection1 = np.asarray([
                get_argument(p1[0], p3[0], t1),
                get_argument(p1[1], p3[1], t1),
            ])
            t2 = get_coefficient(h2, h3)
            intersection2 = np.asarray([
                get_argument(p2[0], p3[0], t2),
                get_argument(p2[1], p3[1], t2),
            ])
            
            edges.append(np.asarray([intersection1, intersection2]))
        return edges


    def sketch_coastline(self):
        # TODO: Seperate points into individual landmasses
        coastal_points = self.get_coastal_points()
        landmasses = self.seperate_landmasses(coastal_points)
        coastlines = [self.calculate_sea_boundary(landmass) for landmass in landmasses]

        for coastline in coastlines:
            for edge in coastline:
                plt.plot(edge[:,0], edge[:, 1], 'k-', linewidth=0.3)
        plt.show()
        # TODO: For each landmass, generate polygon

        # TODO: Fractal subdivide and smooth each coast


    def sketch(self):
        self.setup_sketch()
        self.sketch_coastline()
        # plt.show()

if __name__ == "__main__":
    heightmap = HeightMap(2000, sea_level=0.4)
    heightmap.add_global_gradient()
    # heightmap.add_cone()
    for _ in range(10):
        heightmap.add_hill()
    for _ in range(7):
        heightmap.add_hill(valley=True)
    heightmap.simulate_fluvial_erosion()
    heightmap.simulate_fluvial_erosion()
    heightmap.simulate_fluvial_erosion()


    sketcher = Sketcher(heightmap)
    sketcher.sketch()