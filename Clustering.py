from abc import ABC, abstractmethod
from copy import copy
import enum
import numpy as np


class InitializationMethod(ABC):
    def __init__(self, points:np.ndarray, classes:np.ndarray, n_clusters:int) -> None:
        self.points = points
        self.classes = classes
        self.n_clusters = n_clusters

    @abstractmethod
    def generate():
        pass

class ForgyInitialization(InitializationMethod):
    def generate(self):
        indices = np.random.choice(self.points.shape[0], self.n_clusters)
        return self.points[indices]

class RandomPartitionInitialization(InitializationMethod):
    def generate(self):
        return np.array([np.mean(self.points[self.classes==i],axis=0) for i in range(self.n_clusters)])

class Clustering:

    def __init__(self, points:np.ndarray, n_clusters:int = 2, init_method:InitializationMethod = ForgyInitialization) -> None:
        """
        A class for clustering a set of points into K different clusters based on their proximity

        Parameters
        ----------
        points: ndarray
            The points to cluster, given as a a 2D list of coordinates
        
        n_clusters, int (optional)
            The number of clusterings to make
        """

        # Use random points if no points are given
        self.points = points
        self.N = len(self.points)
        
        self.classes = np.random.choice(range(n_clusters), self.N)
        
        self.centroids = init_method(self.points, self.classes, n_clusters).generate()
        #self.centroids = np.array([np.mean(self.points[self.classes==i],axis=0) for i in range(n_clusters)])
        self.prev_centroids = np.array(self.centroids)


    def cluster(self, max_iter:int = 20):
        """
        Attempt to cluster the given points

        Parameters
        ----------
        max_iter, int
            How many iterations to run before giving up

        Returns
        -------
        points, np.ndarray
            The points given
        
        classes, np.ndarray
            The class index for every point
        
        centroids
            The 2D coordinates of the centroid positions
        """
        print(self.centroids)
        last_classes = copy(self.classes)

        iter:int = 0
        while iter < max_iter:

            # Assignment points the closest centroid
            for i, p in enumerate(self.points):
                
                dists = [np.linalg.norm(p - c) for c in self.centroids]
                self.classes[i] = np.argmin(dists)

            # Update centroid placements
            for i in range(len(self.centroids)):
                class_points = self.points[self.classes==i]

                # Guard against taking the mean of nothing in cases where a class has no points assigned to it
                if 0 < len(class_points):
                    self.centroids[i] = np.mean(self.points[self.classes==i],axis=0)
                
            # Break the loop if the cluster assignment hasnt changed
            if np.array_equal(self.classes, last_classes):
                break
            
            last_classes = copy(self.classes)
            iter += 1
            print(iter)

        return self.points, self.classes, self.centroids
