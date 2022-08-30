import numpy as np
from scipy.spatial import distance

def should_continue(sse_diff, thresh):
    """
    Returns true if we should continue to move our
    centroids.
    
    Args:
        sse_diff  the difference between two consecutive
                  rounds of moving the centroids
        thresh    the change in SSE around zero that
                  indicates a stopping condition
    """
    return sse_diff > thresh


def get_sse(obs, centroids, labels):
    """
    Finds the sum of the square error (distance) between
    centroids and its labeled observations
    """
    
    # calculate distance between all centroids and all observations
    dists = distance.cdist(obs, centroids, metric="euclidean")

    # identify minimum distances, this is distance from labels to their centroids
    min_dists = np.min(dists, axis=1)

    # calculcate sse as sum of squared distances
    sse = round(np.sum(min_dists**2),2)

    return sse


def find_labels(obs, centroids):
    """
    Labels each observation in df based upon
    the nearest centroid. 
    
    Args:
        obs  numpy array of n observations with m dimensions
        centroids numpy array of k centroids with m dimensions
    
    Returns:
        a numpy array of labels, one for each observation
    """
    
    # create array of distances between each point in obs and each centroid
    dist_arr = distance.cdist(obs, centroids, metric="euclidean")

    # find column number of min distance for each obs, corresponds to centroid
    labels = [np.where(x==min(x))[0] for x in dist_arr]

    # create array of labels for each obs
    # if an obs is equidistant from > 1 centroid, only assign first centroid label
    labels_arr = np.array([])
    for z in labels:
        labels_arr = np.append(labels_arr, int(z[0]))

    # convert label values from float to int
    labels_arr = labels_arr.astype("int")

    return labels_arr

        


    
def recompute_centroids(obs, centroids, labels):
    """
    Find the new location of the centroids by
    finding the mean location of all points assigned
    to each centroid
    
    Arguments:
        obs  numpy array of observations with n observations in m dimensions
        centroids  k centroids of m dimensions
        labels  n labels; one for each observation
    
    Returns:
        numpy array of k centroids with m dimensions (updated)
    """

    # initialize list to store new centroid values 
    new_centroids = []

    # loop over current centroids
    for i, c in enumerate(centroids):
        
        # retrieve index values of cluster member points
        cluster_member_indexes = np.where(labels == i)[0]

        # creates array of only cluster members
        cluster_members = np.take(obs, cluster_member_indexes, axis=0)

        # find average of cluster
        clust_avg = np.average(cluster_members, axis=0)
        clust_avg = np.reshape(clust_avg, [1,len(clust_avg)])

        new_centroids.append(clust_avg)

    # convert centroids list to array and reshape
    new_centroids_arr = np.array(new_centroids)
    new_centroids_arr = new_centroids_arr.reshape([len(centroids), len(obs[0])])

    return new_centroids_arr
        

    
    
def cluster_kmeans(obs, k):
    """
    Clusters the n observations of m attributes 
    
    Euclidean distance is used as the proximity metric.
    
    Arguments:
        obs   numpy array of n observations of m dimensions
        k    the number of clusters to search for
        
    Returns:
        a n-sized numpy array of the cluster labels
        
        the final Sum-of-Error-Squared (SSE) from the clustering

        a k x m numpy array of the centroid locations
    """
    
    # randomly select initial centroids
    centroid_index = list(np.random.choice(range(len(obs)), size=k))
    new_centroids = np.array([])
    new_centroids = [np.append(new_centroids, obs[i]) for i in centroid_index]

    # find first labels
    labels = find_labels(obs, new_centroids)

    # find first sse
    new_sse = get_sse(obs, new_centroids, labels)

    carry_on = True
    # loop to recompute centroids until delta(sse) threshold is met
    while carry_on == True:

        # store sse from last run
        old_sse = new_sse

        # compute new centroids
        new_centroids = recompute_centroids(obs, new_centroids, labels)

        # find new labels
        labels = find_labels(obs, new_centroids)

        # calculcate new sse
        new_sse = get_sse(obs, new_centroids, labels)

        sse_delta = new_sse - old_sse

        carry_on = should_continue(sse_diff=sse_delta, thresh=0.01)

    
    return labels, new_sse, new_centroids
