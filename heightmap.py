import numpy as np 
import random
from generate  import generate_mesh
from scipy.spatial import ConvexHull

class HeightMap:
    def __init__(self, npoints, sea_level=0):
        self.size = npoints
        self.sea_level = sea_level
        self.mesh, self.neighbours, self.find_index  = generate_mesh(self.size, 5)
        self.heights = np.random.rand(self.size)*0.1#np.ones(self.size)
        self.border_indices = np.asarray(list(set([ index for edge in ConvexHull(self.mesh).simplices for index in edge])))
        self.flux = None
        self.slope = None
        self.erosion = None

    def normalize(self):
        self.heights -= self.heights.min()
        if self.heights.max() != 0:
            self.heights /= self.heights.max()

    def add_global_gradient(self, angle=None, m=None):
        print('Generating global gradient')
        if angle is None:
            angle = 2*np.pi*random.random()
        if m is None:
            m = random.random()

        # Construct an arbitrary rotation of the points
        xs, ys = self.mesh[:,0] - 0.5, self.mesh[:,1] - 0.5
        radii = np.sqrt(np.power(xs, 2) + np.power(ys, 2))
        phi_primes = np.arctan2(ys, xs) - angle
        xprimes = np.multiply(radii, np.cos(phi_primes))

        # Assume that the points have been rotated to lie along an x axis slope
        self.heights += random.random()*xprimes
        self.normalize()

    def add_hill(self, steepness=None, valley=False, peak_index=None):
        print(f'Generating {"valley" if valley else "hill"}')
        initial_height = random.random()
        current_height = initial_height

        if steepness is None:
            steepness = random.uniform(0.6, 0.8)
        
        height_adjustments = np.zeros(self.size)
        
        # Pick a peak
        if peak_index is None:
            indexes = [random.randint(0, self.size-1)]
        else:
            indexes = [peak_index]

        adjusted = set()

        while current_height / initial_height >= 0.2 and indexes:
            # Set height
            for index in indexes:
                height_adjustments[index] = current_height
                adjusted.add(index)

            # Reduce height
            current_height *= steepness

            # Get new indexes
            new_indexes = []
            for i in indexes:
                neighbours = self.neighbours(self.mesh[i])
                neighbours_indexes = [self.find_index(n) for n in neighbours]
                stripped = [n for n in neighbours_indexes if not n in adjusted]
                new_indexes += stripped
            indexes = new_indexes

        if valley:
            self.heights -= height_adjustments
        else:    
            self.heights += height_adjustments
    
    def add_cone(self, inverted=False):
        # Find point nearest to middle
        middle_index = np.argmin(np.abs(self.mesh - np.asarray([0.5, 0.5])).sum(axis=1))
        self.add_hill(steepness=0.88, valley=inverted, peak_index=middle_index)

    def remove_sinks(self):
        self.normalize()
        # Find exterior vertex indices and construct "fitted sheet"
        sheet = np.asarray([np.inf for _ in self.mesh])
        sheet[self.border_indices] = self.heights[self.border_indices]
        
        undepleted = np.array([True for _ in sheet])
        while np.inf in sheet:
            # Select a known point that has not been depleted
            candidates = np.logical_and(np.less(sheet, np.inf), undepleted)
            lowest_index = np.argmin(sheet[candidates])
            known = self.mesh[candidates][lowest_index]

            # Find unknown linked to edge
            for target_point in self.neighbours(known):
                index = self.find_index(target_point)
                if sheet[index] == np.inf:
                    break
            else:
                # Point is depleted, find it's true index and flag it
                true_index = self.find_index(known)
                undepleted[true_index] = False
                continue

            # Determine height
            neighbours = self.neighbours(target_point)
            n_indexes = [self.find_index(n) for n in neighbours]
            smallest_n_heights = sheet[n_indexes].min() + 0.01 # 0.01 to ensure no flat spots
            sheet[index] = max(smallest_n_heights, self.heights[index])
        self.heights = sheet

    def simulate_fluvial_erosion(self):
        print('Simulating runoff erosion')
        self.remove_sinks()
        self.normalize()
        self.flux = np.ones(self.size)
        self.slope = np.zeros(self.size)

        # Sort from highest to lowest
        sort_keys = np.argsort(self.heights)[::-1]
        for index in sort_keys:
            influx = self.flux[index]
            current_point = self.mesh[index]
            neighbours = self.neighbours(current_point)
            neighbour_indices = [self.find_index(n) for n in neighbours]

            # Compute slope
            dheight = self.heights[neighbour_indices] - self.heights[index]
            dpoints = neighbours - current_point
            distances = np.sqrt(np.sum(np.multiply(dpoints, dpoints), axis=1))
            naive_slope = np.divide(dheight, distances) 

            # Compute and save scalar slope 
            self.slope[index] = np.sum(np.abs(naive_slope)) / len(naive_slope)

            # Compute normalized downhill only flow
            negative_slope = naive_slope
            negative_slope[np.greater(naive_slope, 0)] = 0
            if sum(negative_slope):
                # Last point / sinks will have sum(naive_slope) = 0
                negative_slope = negative_slope / sum(negative_slope)
            outflux = influx * negative_slope

            # Apply flow to other graph nodes
            for i, flow in enumerate(outflux):
                self.flux[neighbour_indices[i]] += flow


        # Compute erosion coefficeints with clamping
        self.erosion = np.sqrt(self.flux) * self.slope + 0.05 * np.power(self.slope, 2)
        self.erosion[np.greater(self.erosion, 0.3)] = 0.3

        # Normalize flux and slope for plotting
        self.flux /= self.flux.max()
        self.slope /= self.slope.max()


        # Apply ersosion
        self.heights -= self.erosion
        
        # self.sea_level -= np.median(self.erosion)*0.3
        self.sea_level = (self.heights.max() - self.heights.min())*0.4 + self.heights.min()
        
        # Clean up isolated points
        for index, point in enumerate(self.mesh):
            underwater = self.heights[index] < self.sea_level
            neighbour_indices = np.asarray([self.find_index(n) for n in self.neighbours(point)])
            neighbours_underwater = np.less(self.heights[neighbour_indices], self.sea_level)

            no_underwater_neighbours = np.count_nonzero(neighbours_underwater)

            if not underwater:
                if no_underwater_neighbours / len(neighbour_indices) > 0.5:
                    underwater_mean_height = np.mean(
                        self.heights[neighbour_indices[neighbours_underwater]]
                    )
                    self.heights[index] = underwater_mean_height
            if underwater:
                no_above_water_neighbours = len(neighbour_indices) - no_underwater_neighbours
                if no_above_water_neighbours / len(neighbour_indices) > 0.5:
                    above_water_mean_height = np.mean(
                        self.heights[neighbour_indices[np.logical_not(neighbours_underwater)]]
                    )
                    self.heights[index] = above_water_mean_height


    def show_height(self, lines=False, style='.', save=False, name=''):
        print('Plotting heightmap')

        from matplotlib import pyplot as plt 

        for i, p in enumerate(self.mesh):
            height = self.heights[i]
            if height > self.sea_level:
                height -= self.heights.min()
                height /= self.heights.max()
                height = max(0, min(1, height))
                col = (height, 0, 1-height, 0.8)
                plt.plot(p[0], p[1], style, color=col)
                if lines:
                    neighbours = self.neighbours(p)
                    for n in neighbours:
                        plt.plot([p[0], n[0]], [p[1], n[1]], 'k-', linewidth=0.1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        if save == True:
            plt.savefig(f'{name}.png')
        else:
            plt.show()
        plt.clf()

    def show_height_interactive(self, lines=False, style='.'):
        print('Plotting heightmap')

        from matplotlib import pyplot as plt 
        from matplotlib.widgets import Slider


        sea_level = self.sea_level
        axplot = plt.axes([0.25, 0.13, 0.65, 0.8])
        axslider = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(axslider, 'Sea level', 0, 1, valinit=self.sea_level)

        def update(val):
            axplot.cla()
            axplot.set_xlim([0, 1])
            axplot.set_ylim([0, 1])
            sea_level = slider.val
            for i, p in enumerate(self.mesh):
                height = self.heights[i]
                if height > sea_level:
                    col = (height, 0, 1-height, 0.8)
                    axplot.plot(p[0], p[1], style, color=col)
                    if lines:
                        neighbours = self.neighbours(p)
                        for n in neighbours:
                            axplot.plot([p[0], n[0]], [p[1], n[1]], 'k-', linewidth=0.1)

        slider.on_changed(update)
        update(None)
        plt.show()
        plt.clf()

    def show_flux(self, lines=False, style='.'):
        print('Plotting fluxmap')
        assert not self.flux is None

        from matplotlib import pyplot as plt 
        for i, p in enumerate(self.mesh):
            flux = self.flux[i]
            col = (flux, 0, 1-flux, 0.8)
            plt.plot(p[0], p[1], style, color=col)
            if lines:
                neighbours = self.neighbours(p)
                for n in neighbours:
                    plt.plot([p[0], n[0]], [p[1], n[1]], 'k-', linewidth=0.1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.show()
        plt.clf()

    def show_slope(self, lines=False, style='.'):
        print('Plotting slopemap')
        assert not self.slope is None

        from matplotlib import pyplot as plt 
        for i, p in enumerate(self.mesh):
            slope = self.slope[i]
            col = (slope, 0, 1-slope, 0.8)
            plt.plot(p[0], p[1], style, color=col)
            if lines:
                neighbours = self.neighbours(p)
                for n in neighbours:
                    plt.plot([p[0], n[0]], [p[1], n[1]], 'k-', linewidth=0.1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.show()
        plt.clf()

    def show_erosion(self, lines=False, style='.'):
        print('Plotting erosionmap')
        assert not self.erosion is None

        from matplotlib import pyplot as plt 
        for i, p in enumerate(self.mesh):
            erosion = self.erosion[i]
            col = (erosion, 0, 1-erosion, 0.8)
            plt.plot(p[0], p[1], style, color=col)
            if lines:
                neighbours = self.neighbours(p)
                for n in neighbours:
                    plt.plot([p[0], n[0]], [p[1], n[1]], 'k-', linewidth=0.1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.show()
        plt.clf()
# TODO: Replace discrete random with continuous random
# TODO: Add a smooth function

if __name__ == "__main__":
    heightmap = HeightMap(300, sea_level=0.4)
    heightmap.add_global_gradient()
    # heightmap.add_cone()
    for _ in range(10):
        heightmap.add_hill()
    for _ in range(7):
        heightmap.add_hill(valley=True)

    heightmap.show_height(save=True, name=0)
    for i in range(1, 3):
        heightmap.simulate_fluvial_erosion()
        heightmap.show_height(save=True, name=i)

    