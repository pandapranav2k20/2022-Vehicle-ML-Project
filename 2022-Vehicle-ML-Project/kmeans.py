
"""
@author: pranavveerubhotla

This algorithm preforms k-means clustering using random initialization of
    cluster locations.
"""
import numpy as np
import math
from sklearn.metrics import pairwise_distances_argmin
def kmeans(X, random_state, n_clusters=3, tol=1e-3, max_iter=100):
    ''' 
    (1) Randomly Select the Clusters
    '''
    random_num_gen = np.random.RandomState(random_state)
    r_ind = random_num_gen.permutation(X.shape[0])[:n_clusters]
    #print(r_ind)
    cluster_centers = X[r_ind]
    initial_cluster_centers = cluster_centers
    #print(cluster_centers)
    n_iter = 1
    all_WCSS = []
    while True:
        '''
        (2) Assign Cluster Labels to ALL Points in X
        '''
        c_labels = pairwise_distances_argmin(X, cluster_centers)
        points_organized = []       
        for g in range(0, n_clusters):
            current_cluster = []
            for h in range(0, len(c_labels)):
                if(c_labels[h] == g):
                    current_cluster.append(X[h])
            points_organized.append(current_cluster)
        '''
        (3) Compute the Within Cluster Sum of Squares [WCSS]
        '''
        WCSS = 0.0
        for j in range(0, n_clusters):
            #print(points_organized[j], '\n')
            CSS = 0.0
            for f in points_organized[j]:
                CSS += (math.hypot(f[0] - cluster_centers[j][0], 
                                   f[1] - cluster_centers[j][1]))**2
            #print('Cluster: ', j, 'CSS =',CSS)
            WCSS += CSS
        all_WCSS.append(WCSS)
        #print('Iteration: ', n_iter, 'WCSS =', WCSS)
        '''
        (4) Locate the New Cluster Centers By Using the Means of the Clusters
        '''
        new_cluster_centers = np.array([X[c_labels == i].mean(0)
                                for i in range(n_clusters)])
        '''
        (5) Determine Whether to Continue or Break While loop
        '''
        if (n_iter == 1):
            cluster_centers = new_cluster_centers
            past_WCSS = WCSS
            n_iter += 1
            continue
        else:
            #WCSS_delta = abs(all_WCSS[n_iter - 2] - all_WCSS[n_iter - 3])
            WCSS_delta = abs(WCSS-past_WCSS)
            if ((WCSS_delta > tol) and (n_iter < max_iter)):
                cluster_centers = new_cluster_centers
                past_WCSS = WCSS
                n_iter += 1
                continue
            else:
                '''CONDITION TO STOP ITERATING'''
                n_iter += 1
                break
    return c_labels, cluster_centers, initial_cluster_centers