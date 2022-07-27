from copy import copy

import matplotlib.pyplot as plt
import numpy as np

# https://matplotlib.org/stable/gallery/color/named_colors.html
colors = np.array(['dodgerblue', 'gold', 'crimson', 'forestgreen'])

class Clustering:

    def __init__(self, points:np.ndarray = None, N:int = 50, n_clusters:int = 3) -> None:

        if points:
            self.points = points
        else:
            self.points = self.__random_points(N)

        self.N = len(self.points)        
        self.classes = np.zeros(self.N)
        self.n_clusters = n_clusters
        self.centroids = np.random.rand(n_clusters, 2)
        self.__configure_plot()
        

    def __random_points(self, N:int) -> np.ndarray:
        return np.random.rand(N, 2)

    def __configure_plot(self, figsize:tuple=(7, 7)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim([0,1])
        self.ax.set_ylim([0,1])


    def __plot_points(self, ax, p, c=None, marker='+'):
        if c:
            ax.scatter(x=p[:,0], y=p[:,1], c=c, marker=marker)
        else:
            ax.scatter(x=p[:,0], y=p[:,1], c='black', marker=marker)
        

    def __plot_step(self):
        # self.ax.clear()
        self.__plot_points(self.ax, self.points, marker='+')
        self.__plot_points(self.ax, self.centroids, range(self.n_clusters), marker='o')
        
        plt.show()
        

    def cluster(self, plot_steps:bool = False):
        last_classes = copy(self.classes)

        while True:

            # Assignment step
            for i, p in enumerate(self.points):
                
                dists = [np.linalg.norm(p - c) for c in self.centroids]
                self.classes[i] = np.argmin(dists)
                

            # Update step
            for i in range(len(self.centroids)):
                self.centroids[i] = np.mean(self.points[self.classes==i],axis=0)

            if plot_steps:
                print("here")
                self.__plot_step()

            if np.array_equal(self.classes, last_classes):
                break
            else:
                last_classes = copy(self.classes)

        return self.points, self.classes, self.centroids
