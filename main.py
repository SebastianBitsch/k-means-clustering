from Clustering import Clustering
import numpy as np
import matplotlib.pyplot as plt

def generate_random_cloud(bounds:list[tuple] = [(0,0),(5,5)]) -> np.ndarray:
    n = np.random.randint(20,100)
    center = bounds[0] + np.random.rand(2) * bounds[1]
    cov = np.random.rand(2,2) # [[0.01,0],[0,0.01]] #
    
    return multivariate_normal_points(n, mu=center, cov=cov)


def multivariate_normal_points(N:int, mu:np.ndarray = [0.5,0.5], cov:np.ndarray = [[0.01,0],[0,0.01]]) -> np.ndarray:
    return np.random.multivariate_normal(mu, cov, N)


if __name__ == "__main__":  
    
    n_clusters = 3
    bounds = [(0,0),(15,15)]
    
    clouds = [generate_random_cloud(bounds) for _ in range(n_clusters)]
    points = np.vstack(clouds)

    c = Clustering(points, n_clusters)

    # points, classes, centroids = c.cluster()
    # plt.scatter(x=points[:,0],y=points[:,1],c=classes)
    # plt.scatter(x=centroids[:,0],y=centroids[:,1],c='black')
    # plt.show()    
