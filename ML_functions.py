# ML_functions
"""
Contains different Machine Learning functions, like

"""

import tensorflow as tf
from photonpy import PostProcessMethods, Context
import tensorflow.math as mth


def vicinity_neighbours(ch1, ch2, maxDistance):
    '''
    Finds the neighbours between ch1 and ch2 for a maxDistance

    Parameters
    ----------
    ch1, ch2 : 2xN float array
        Contains the [x1,x2] locations of the localizations in ch1 and ch2.
    maxDistance : float32
        The maximum distance that the Neighbours can be in.

    Returns
    -------
    neighbours : Nx2 float array
        An array containing in the first column the localization index in ch1 
        and in the second column the localization index of its neighbours in ch2.

    '''
    with Context() as ctx:
        counts,indices = PostProcessMethods(ctx).FindNeighbors(
        tf.transpose(ch1) , tf.transpose(ch2), maxDistance=maxDistance)
    
    neighbours_ch1 = tf.Variable([], tf.int32)
    neighbours_ch2 = tf.Variable([], tf.int32)
    N = ch1.shape[1]
    j1 = 0
    for i in range(N):        
        # Find the indices of the neighbours of localization i
        j2 = 0
        while j2 < counts[i]:    
            neighbours_ch1 = tf.concat([ neighbours_ch1, [int(indices[j1])] ], axis = 0)
            neighbours_ch2 = tf.concat([ neighbours_ch2, [int(i)] ], axis = 0)
            j2 += 1
            j1 += 1
         
    #neighbours = tf.slice(neighbours, 1, neighbours.shape[0])
    return neighbours_ch1, neighbours_ch2
        
        
def nearest_neighbours(ch1, ch2, maxDistance, k):
    with Context() as ctx:
        counts,indices = PostProcessMethods(ctx).FindNeighbors(
        tf.transpose(ch1) , tf.transpose(ch2), maxDistance=maxDistance)
    
    neighbours = []
    N = ch1.shape[1]
    j1 = 0
    for i in range(N):    
        temp_j = []
        # Find the indices of the neighbours of localization i
        j2 = 0
        while j2 < counts[i]: 
            temp_j.append([i, indices[j1]])
            j2 += 1
            j1 += 1
            
        if len(temp_j) > k:
            # we are only interested in the k closest values
            temp_r = mth.sqrt(
                ( ch1[0, temp_j] - ch2[0, temp_j] )**2 +
                ( ch1[1, temp_j] - ch2[1, temp_j] )**2
                )
            
            k_smallest = tf.argsort(temp_r)[0:k]
            neighbours.append( temp_j[ k_smallest , : ] )            
        else:
            neighbours.append( temp_j )
        
    return neighbours