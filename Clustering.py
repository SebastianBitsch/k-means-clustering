from InitializationMethods import InitializationMethod, ForgyInitialization, RandomPartitionInitialization

from copy import copy

import numpy as np


class Clustering:

    def __init__(self, points:np.ndarray, n_clusters:int = 2, init_method:InitializationMethod = RandomPartitionInitialization) -> None:
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
        
        self.classes = [None for _ in range(self.N)]#np.random.choice(range(n_clusters), self.N) # TODO: should probably be set after/using the centroids
        
        self.init_method = init_method(self.points, self.classes, n_clusters)


    def cluster(self, runs:int = 1, max_iter:int = 100):
        """
        Attempt to cluster the given points

        Parameters
        ----------
        runs, int = 1
            How many times to run the algorithm in succession, classes for points are then voted on
            More attempts help remove dependence on the initial clustering

        max_iter, int = 100
            The maximum number of iterations to complete before returning the clustering

        Returns
        -------
        classes, np.ndarray
            The class index for every point
        
        centroids
            The 2D coordinates of the centroid positions
        """

        classes_total = []
        centroids_total = []

        for i in range(runs):
            # Generate new init conditions
            centroids = self.init_method.generate()
            classes, centroids = self._run_clustering_single(self.points, centroids, max_iter=max_iter)
            
            classes_total.append(classes)
            centroids_total.append(centroids)

        # Do voting - one problem is that [1,1,2,2] should be the same as [2,2,1,1]
        # np.argmax(np.bincount(total_classes[:,0])) - to find most comment element in array

        return classes, centroids

    def _run_clustering_single(self, points:np.ndarray, centroids:np.ndarray, max_iter:int):
    
        classes = np.full(self.N, None)
        last_classes = np.full(self.N, None)

        iteration:int = 0
        while iteration < max_iter:

            # Assignment points the closest centroid
            for i, p in enumerate(points):                
                dists = [np.linalg.norm(p - c) for c in centroids]
                classes[i] = np.argmin(dists)

            # Update centroid placements
            for i in range(len(centroids)):
                centroids[i] = np.mean(points[classes==i],axis=0)

            # Break the loop if the cluster assignment hasnt changed
            if np.array_equal(classes, last_classes):
                break
            
            last_classes = copy(classes)
            iteration += 1

        return classes, centroids