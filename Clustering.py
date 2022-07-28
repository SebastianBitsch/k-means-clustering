from copy import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore") # Hide a useless matplotlib warning

# https://matplotlib.org/stable/gallery/color/named_colors.html
colors = np.array(['dodgerblue', 'gold', 'crimson', 'forestgreen', 'indigo', 'teal', 'orange', 'palevioletred'])


def random_points(N:int, bounds:tuple = (1,1)) -> np.ndarray:
    return np.random.rand(N, 2) * bounds

def bound_points(points:np.ndarray, bounds:tuple = (1,1)):
    return points[np.logical_and(np.logical_and(0 < points[:,0], points[:,0] < bounds[0]), np.logical_and(0 < points[:,1], points[:,1] < bounds[1]))]



class Clustering:

    def __init__(self, points:np.ndarray = None, N:int = 50, n_clusters:int = 3, bounds:tuple = (1,1)) -> None:

        self.points = random_points(N, bounds) if points is None else points
        self.points = bound_points(self.points, bounds)
        
        self.N = len(self.points)
        self.bounds = bounds
        
        self.classes = np.zeros(self.N, dtype=int)
        self.centroids = random_points(n_clusters, bounds)
        self.prev_centroids = np.array([copy(self.centroids)])
        


    def __plot_points(self, ax:plt.Axes, p:list, c='black', marker:str='+', opacity:float=1):
        ax.scatter(x=p[:,0], y=p[:,1], facecolor=c, marker=marker, alpha=opacity, edgecolors='black', s=80)
    
    
    def __plot_initial_setup(self, figsize:tuple = (7,7), show_cluster_centers:bool = False, title:str = ""):
        _, ax = plt.subplots(figsize=figsize)
        plt.title(title, loc='left', fontweight='bold')

        ax.set_xlim([0, self.bounds[0]])
        ax.set_ylim([0, self.bounds[1]])
        
        # Plot points and cluster centers
        self.__plot_points(ax=ax, p=self.points, c='black', marker='+')
        if show_cluster_centers:
            self.__plot_points(ax=ax, p=self.centroids, c=colors[np.arange(len(self.centroids))], marker='o')
        plt.show()


    def __plot_step(self, figsize:tuple = (7,7), title:str = ""):
        _, ax = plt.subplots(figsize=figsize)
        plt.title(title, loc='left', fontweight='bold')
        

        ax.set_xlim([0, self.bounds[0]])
        ax.set_ylim([0, self.bounds[1]])
        
        # Plot points and cluster centers
        self.__plot_points(ax=ax, p=self.points, c=colors[self.classes], marker='+')
        
        # Plot the trail of cluster centers with decreasing opacity
        for i, c in enumerate(self.prev_centroids):
            opacity = 1 / (len(self.prev_centroids) - i)
            self.__plot_points(ax=ax, p=c, c=colors[np.arange(len(self.centroids))], marker='o', opacity=opacity)

            if i+1 == len(self.prev_centroids):
                continue

            c1 = self.prev_centroids[i+1]
            for j in range(len(self.centroids)):
                ax.plot((c[j][0], c1[j][0]), (c[j][1], c1[j][1]), c='lightgray', alpha=1, zorder=-8)
            

        # Plot lines from cluster to every point
        for i, p in enumerate(self.points):
            center = self.centroids[self.classes[i]]
            ax.plot((p[0], center[0]), (p[1], center[1]), c='lightgray', alpha=0.2, zorder=-10)


        plt.show()


    def cluster(self, plot_steps:bool = False):
        last_classes = copy(self.classes)

        if plot_steps:
            self.__plot_initial_setup(show_cluster_centers=False, title="1: Initialize random points")
            self.__plot_initial_setup(show_cluster_centers=True, title="2: Initialize random cluster centers")

        while True:

            # Assignment points the closest centroid
            for i, p in enumerate(self.points):
                
                dists = [np.linalg.norm(p - c) for c in self.centroids]
                self.classes[i] = np.argmin(dists)

            if plot_steps:
                self.__plot_step(title="3: Assign points to closest cluster center")


            # Update centroid placements
            for i in range(len(self.centroids)):
                class_points = self.points[self.classes==i]

                # Guard against taking the mean of nothing in cases where a class has no points prescribed to it
                if 0 < len(class_points):
                    self.centroids[i] = np.mean(self.points[self.classes==i],axis=0)
                
            # Plot the step
            if plot_steps:
                self.prev_centroids = np.append(self.prev_centroids,[copy(self.centroids)],axis=0)
                self.__plot_step(title="4: Move centroid center to the middle of the points")

            # Break the loop if the cluster assignment hasnt changed
            if np.array_equal(self.classes, last_classes):
                break
            else:
                last_classes = copy(self.classes)

        if plot_steps:
            self.__plot_step(title="5: A stable configuration has been found")

        return self.points, self.classes, self.centroids
