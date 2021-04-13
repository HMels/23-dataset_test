# ML_functions
"""
Contains different Machine Learning functions, like

"""

import tensorflow as tf
#from photonpy import PostProcessMethods, Context
import numpy as np

def KNN(ch1, ch2, k):
    '''
    k-Nearest Neighbour Distance calculator

    Parameters
    ----------
    ch1, ch2 : 2D float32 array
        The array containing the [x1, x2] locations of the localizations i and j.
    sigma_i, sigma_j : 2D float32 array
        The array containing the [x1, x2] std of the localizations i and j.
    k : int
        The number of kNN the KL-Divergence should be calculated for

    Returns
    -------
    knn : [k, N, 2] TensorFlow Tensor
        Tensor Containing the squared [x1,x2] distances for k rows of kNN, 
        for all N localizations in the colums.

    '''
    N_locs = ch1.shape[0]
    # distances contains all distances with the second axis being the distances to ch1
    # and the third axis being the [x1,x2] axis
    
    distances = tf.square(tf.add( 
        ch1[None,:,:] , tf.negative( ch2[:,None,:] )
        ))
    abs_distances = tf.reduce_sum(distances, 2)
    
    neg_one = tf.constant(-1.0, dtype=tf.float32)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply( tf.transpose(abs_distances) , neg_one)
    # get the indices
    _, indx = tf.nn.top_k(neg_distances, k)
    
    # getting index in the right format for a gather_nd
    indx = tf.reshape(indx,[1, indx.shape[0]*indx.shape[1] ])
    indx1 = np.linspace(0,  N_locs-1, N_locs, dtype = int) * np.ones([k,1], dtype = int)
    indx1 = tf.reshape(indx1, [1, indx.shape[0]*indx.shape[1] ])
    indx = tf.transpose( tf.stack( [indx1, indx] ) )
    
    knn = tf.reshape( tf.gather_nd( distances, indx ), [k, N_locs, 2] )
    return knn