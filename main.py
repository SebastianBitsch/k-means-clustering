from Clustering import Clustering
import numpy as np


def generate_random_cloud(bounds:tuple = (5,5)):
    n = np.random.randint(20,100)
    center = np.random.rand(2) * bounds
    cov = np.random.rand(2,2)

    return multivariate_normal_points(n, mu=center, cov=cov)


def multivariate_normal_points(N:int, mu:np.ndarray = [0.5,0.5], cov:np.ndarray = [[0.01,0],[0,0.01]]):
    return np.random.multivariate_normal(mu, cov, N)


if __name__ == "__main__":  
    
    n_clusters = 3
    bounds = (15,15)

    clouds = [generate_random_cloud(bounds) for _ in range(n_clusters)]
    points = np.vstack(clouds)

    c = Clustering(points, n_clusters=n_clusters, bounds=bounds)

    c.cluster(plot_steps=True)
