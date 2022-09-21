from Clustering import Clustering
import numpy as np
import matplotlib.pyplot as plt


def random_points(N:int, bounds:tuple = [(0,0),(1,1)]) -> np.ndarray:
    """ Generate N random uniform points within a given set of bounds"""
    return bounds[0] + (np.random.rand(N, 2) * bounds[1])


def multivariate_normal_points(bounds:list[tuple] = [(0,0),(5,5)]) -> np.ndarray:
    N = np.random.randint(20,100)
    mu = bounds[0] + np.random.rand(2) * bounds[1]
    cov = np.random.rand(2,2) # [[0.01,0],[0,0.01]] #
    
    return np.random.multivariate_normal(mu, cov, N)


if __name__ == "__main__":  
    
    n_clusters = 3
    bounds = [(0,0),(15,15)]
    
    clouds = [multivariate_normal_points(bounds) for _ in range(n_clusters)]
    points = np.vstack(clouds)

    c = Clustering(points, n_clusters)

    points, classes, centroids = c.cluster()
    plt.scatter(x=points[:,0],y=points[:,1],c=classes, cmap='Set1')
    plt.scatter(x=centroids[:,0],y=centroids[:,1],c='black',marker='x')
    plt.show()
