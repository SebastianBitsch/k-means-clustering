from copy import copy
import numpy as np

class Clustering:

    def __init__(self, points:np.ndarray = None, n_clusters:int = 3) -> None:
        """
        A class fr clustering a set of points into K different clusters based on their proximity

        Parameters
        ----------
        points: ndarray (optional)
            The points to cluster, given as a a 2D list of coordinates
        
        n_clusters, int (optional)
            The number of clusterings to make
        """

        # Use random points if no points are given
        self.points = points
        self.N = len(self.points)

        self.bounds = self.calculate_bounds(points)

        self.classes = np.zeros(self.N, dtype=int)
        
        self.centroids = self.random_points(n_clusters, self.bounds)
        self.prev_centroids = np.array([copy(self.centroids)])


    def calculate_bounds(self, points:np.ndarray) -> list:
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        return [(min_x,min_y),(max_x,max_y)]


    def random_points(self, N:int, bounds:tuple = [(0,0),(1,1)]) -> np.ndarray:
        """ Generate N random uniform points within a given set of bounds"""
        return bounds[0] + (np.random.rand(N, 2) * bounds[1])


    def cluster(self):
        last_classes = copy(self.classes)

        while True:

            # Assignment points the closest centroid
            for i, p in enumerate(self.points):
                
                dists = [np.linalg.norm(p - c) for c in self.centroids]
                self.classes[i] = np.argmin(dists)

            # Update centroid placements
            for i in range(len(self.centroids)):
                class_points = self.points[self.classes==i]

                # Guard against taking the mean of nothing in cases where a class has no points prescribed to it
                if 0 < len(class_points):
                    self.centroids[i] = np.mean(self.points[self.classes==i],axis=0)
                
            # Break the loop if the cluster assignment hasnt changed
            if np.array_equal(self.classes, last_classes):
                break
            
            last_classes = copy(self.classes)

        return self.points, self.classes, self.centroids
