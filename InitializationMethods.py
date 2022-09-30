from abc import ABC, abstractmethod

import numpy as np


class InitializationMethod(ABC):
    """
    Base class for different implementations of initialization methods for k-means-clustering
    """
    def __init__(self, points:np.ndarray, classes:np.ndarray, n_clusters:int) -> None:
        self.points = points
        self.classes = classes
        self.n_clusters = n_clusters

    @abstractmethod
    def generate():
        pass

class ForgyInitialization(InitializationMethod):
    """ 
    The Forgy method randomly chooses k observations from the dataset and uses these as the 
    initial means. 
    """
    def generate(self):
        indices = np.random.choice(self.points.shape[0], self.n_clusters)
        return self.points[indices]

class RandomPartitionInitialization(InitializationMethod):
    """ 
    The Random Partition method first randomly assigns a cluster to each observation and then 
    proceeds to the update step, thus computing the initial mean to be the centroid of the 
    cluster's randomly assigned points. 
    """
    def generate(self):
        return np.array([np.mean(self.points[self.classes==i],axis=0) for i in range(self.n_clusters)])
