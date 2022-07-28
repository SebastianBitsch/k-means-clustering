from Clustering import Clustering, multivariate_normal_points
import numpy as np

if __name__ == "__main__":

    # c = Clustering(N=500, n_clusters=3, bounds=(3,3))
    p1 = multivariate_normal_points(100, mu=[2,2], cov=[[0.5,0.5],[0.5,0.5]])
    p2 = multivariate_normal_points(100, mu=[3,0.5], cov=[[0.4,0],[0,0.09]])
    p3 = multivariate_normal_points(100, mu=[1,4], cov=[[0.01,0],[0,0.05]])
    points = np.vstack([p1,p2,p3])
    c = Clustering(points, n_clusters=3, bounds=(5,5))
    c.cluster(plot_steps=True)
