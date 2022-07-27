from copy import copy
import matplotlib.pyplot as plt
import numpy as np


colors = np.array(['dodgerblue', 'gold', 'crimson', 'forestgreen'])
# https://matplotlib.org/stable/gallery/color/named_colors.html

# def plot_configuration(points, centroids, classes=None, figsize:tuple=(7,7)) -> None:
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot()

#     if classes:
#         ax.scatter(x=points[:,0], y=points[:,1], c=colors[classes], marker='+')
#     else:
#         ax.scatter(x=points[:,0], y=points[:,1], c='black', marker='+')
    
#     ax.scatter(x=centroids[:,0], y=centroids[:,1], c='black', marker='.')
    
#     ax.set_xlim([0,1])
#     ax.set_ylim([0,1])
#     plt.show()

from Clustering import Clustering

if __name__ == "__main__":
    c = Clustering(N=100)
    c.cluster(plot_steps=True)

    # N = 100
    # n_centroids = 3

    # centroids = np.random.rand(n_centroids, 2)
    # classes = np.zeros(N, dtype=int)
    # last_classes = copy(classes)

    # points = np.random.rand(N, 2)
    
    # plot_configuration(points, centroids, classes)

    # while True:

    #     # Assignment step
    #     for i, p in enumerate(points):
            
    #         dists = [np.linalg.norm(p - c) for c in centroids]
    #         classes[i] = np.argmin(dists)
            

    #     # Update step
    #     for i, c in enumerate(centroids):
    #         centroids[i] = np.mean(points[classes==i],axis=0)
        

    #     if np.array_equal(classes, last_classes):
    #         break
    #     else:
    #         last_classes = copy(classes)

    #     plot_configuration(points, centroids, classes)    
    #     #for i, p in enumerate(points):

    # plot_configuration(points, centroids, classes)