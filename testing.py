import numpy as np
import scipy as sp
import pandas as pd

class WrongNumberOfLabelsError(Exception):
    pass

class LabelNamingIncorrectError(Exception):
    pass


def evaluate(pts, k):
    from testing import do_test, WrongNumberOfLabelsError, LabelNamingIncorrectError
    from kmeans import cluster_kmeans
    num_reps = 10
    try:
        err, mapping, mtx = do_test(cluster_kmeans, num_reps, pts, k)
    except WrongNumberOfLabelsError as err:
        print(err)
        print('Your code is not producing the correct number of labels.')
        raise err
    except LabelNamingIncorrectError as err:
        print(err)
        print('Your code should produce integer labels in the range (0,k-1)')
        raise err

    print(f'Mapping: {mapping}')
    print('Confusion Matrix: ')
    print(mtx)
    print()
    print(f'Error Rate (<=0.05 req\'d): {err}')
    return err



def clustering_error(original, test):
    """
    Given two Pandas Series of labels, find the classification
    error.  This is done by first generating a confusion matrix
    for each label and then classifying points that are not
    the majority for each label as an error.
    
    Args:
        original   Series of original labels
        test       Series of test labels
        
    Returns:
        The fraction of points classified as being in error and
        the original->test mapping as a dictionary.
    """
    original_labels = set(original.unique())
    test_labels = set(np.unique(test))
    if len(test_labels) != len(original_labels):
        msg = 'Original number of labels is {}\n'.format(len(original_labels))
        msg += 'Your code produces {} labels: {}'.format(len(test_labels), test_labels)
        raise WrongNumberOfLabelsError(msg)
    if test_labels != original_labels:
        msg = 'Original number of labels is {}\n'.format(len(original_labels))
        msg += 'Your code produces {} labels: {}'.format(len(test_labels), test_labels)
        raise LabelNamingIncorrectError(msg)
    k = len(original_labels)
    mtx = np.zeros((k,k))
    m = len(original)
    try:
        for i in range(m):
            lab_orig = original[i]
            lab_test = test[i]
            mtx[lab_orig, lab_test] += 1
    except IndexError:
        msg = "lab_orig: {}, lab_test: {}".format(lab_orig, lab_test)
        raise IndexError(msg)
    max_cols = np.max(mtx, axis=1)
    max_ndx = np.argmax(mtx, axis=1)
    sum_error = np.sum(np.sum(mtx, axis=1)-max_cols)
    frac_error = sum_error / m
    label_map = {}
    for i in range(k):
        label_map[i] = max_ndx[i]
    return frac_error, label_map, mtx




def generate_2D_blob(X, Y, n_points, label):
    """
    Generate a 2D blob of points.
    
    Args:
        X          tuple (mean, var) for X-dimension
        Y          tuple (mean, var) for Y-dimension
        n_points   number of points
        label      the label to return
        
    Returns:
        DataFrame with columns [x,y,label]
    """
    mean_x, var_x = X
    mean_y, var_y = Y
    x_pts = np.random.normal(mean_x, var_x, n_points)
    y_pts = np.random.normal(mean_y, var_y, n_points)
    df = pd.DataFrame()
    df['x'] = x_pts
    df['y'] = y_pts
    df['label'] = label
    return df


def generate_3D_blob(X, Y, Z, n_points, label):
    """
    Generate a 3D blob of points.
    
    Args:
        X          tuple (mean, var) for X-dimension
        Y          tuple (mean, var) for Y-dimension
        Z          tuple (mean, var) for Y-dimension
        n_points   number of points
        label      the label to return
        
    Returns:
        DataFrame with columns [x,y,z,label]
    """
    mean_x, var_x = X
    mean_y, var_y = Y
    mean_z, var_z = Z
    x_pts = np.random.normal(mean_x, var_x, n_points)
    y_pts = np.random.normal(mean_y, var_y, n_points)
    z_pts = np.random.normal(mean_z, var_z, n_points)
    df = pd.DataFrame()
    df['x'] = x_pts
    df['y'] = y_pts
    df['z'] = z_pts
    df['label'] = label
    return df


def do_test(func, num_tests, pts, num_clusters):
    """
    Perform num_tests replicatses trying to cluster
    the dataframe pts (m observations, n attributes + 1 label series)
    into num_clusters.

    Note: We're converting the data frame into a numpy array before
    we send it to the clustering function, named func.
    """
    
    min_sse = None  # Will hold the min SSE
    min_sse_labels = None # Will hold the labels for the clustering with min SSE

    # Perform num_tests clusterings
    for k in range(0,num_tests):
        # Below we're converting our data frame into a numpy matrix
        # ignoring the index and last column
        labels, sse, centroids = func(pts.iloc[:,1:-1].to_numpy(),num_clusters)
        if min_sse is None or sse < min_sse:
            min_sse = sse
            min_sse_labels = labels
    
    # Optional -- plotting (can only run on your local machine)
    # do_plot(pts, min_sse_labels, centroids)

    # Calculate its performance
    err,mapping,mtx = clustering_error(pts["label"], min_sse_labels)
    return err,mapping,mtx



def do_plot(pts, labels, centroids):
    """
    A utility function that will call either 2 or 3-D plotting
    methods to plot the clusters (colored by found label) and
    their centroids.

    Arguments:
        pts: A data frame containing m observations with n attributes (2 or 3) and an index
        labels: A numpy array that contains the labels found via clustering
        centroids: A k x n numpy matrix of locations of the found centroids from the clustering
    """
    if centroids.shape[1] == 2:
        do_plot_2D(pts, labels, centroids)
    elif centroids.shape[1] == 3:
        do_plot_3D(pts, labels, centroids)
    else:
        raise NotImplementedError('Cannot plot data with ', centroids.shape[1], ' dimensions')

def do_plot_2D(obs, found, centroids):
    """
    Make a 2-D plot of 2-dimensional clustering output
    See do_plot for details.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.scatterplot(x=obs.iloc[:,1], y=obs.iloc[:,2], hue=found)
    sns.scatterplot(x=centroids[:,0], y=centroids[:,1], color='black', marker='x', s=48)
    plt.show()
    

def do_plot_3D(obs, found, centroids):
    """
    Make a 3-D plot of 3-dimensional clustering output
    See do_plot for details.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(sns.color_palette("tab10", 256).as_hex())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(obs.iloc[:,1], obs.iloc[:,2], obs.iloc[:,3], c=found, cmap=cmap)
    ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], color='black', marker='x', s=48)
    plt.show()
    return