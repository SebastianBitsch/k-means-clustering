from copy import copy
from xml.sax.handler import all_properties

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
        self.classes = np.zeros(self.N, dtype=int)
        self.n_clusters = n_clusters
        self.centroids = np.random.rand(n_clusters, 2)
        self.prev_centroids = np.array([copy(self.centroids)])
        

    def __random_points(self, N:int) -> np.ndarray:
        return np.random.rand(N, 2)

    def __plot_points(self, ax:plt.Axes, p:list, c='black', marker:str='+', opacity:float=1):
        ax.scatter(x=p[:,0], y=p[:,1], c=c, marker=marker, alpha=opacity)
        
        

    def __plot_step(self, figsize:tuple = (8,8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        
        # Plot points and cluster centers
        self.__plot_points(ax=ax, p=self.points, c=colors[self.classes], marker='+')
        
        # Plot the trail of cluster centers with decreasing opacity
        for i, c in enumerate(self.prev_centroids):
            opacity = 1 / (len(self.prev_centroids) - i)
            self.__plot_points(ax=ax, p=c, c=colors[np.arange(self.n_clusters)], marker='o', opacity=opacity)
        
        plt.show()

    def cluster(self, plot_steps:bool = False):
        last_classes = copy(self.classes)

        while True:

            # Assignment step
            for i, p in enumerate(self.points):
                
                dists = [np.linalg.norm(p - c) for c in self.centroids]
                self.classes[i] = np.argmin(dists)

            # Update centroids
            for i in range(len(self.centroids)):
                self.centroids[i] = np.mean(self.points[self.classes==i],axis=0)

            # Plot the step
            if plot_steps:
                self.prev_centroids = np.append(self.prev_centroids,[copy(self.centroids)],axis=0)
                self.__plot_step()

            # Break the loop if the cluster assignment hasnt changed
            if np.array_equal(self.classes, last_classes):
                break
            else:
                last_classes = copy(self.classes)

        return self.points, self.classes, self.centroids
