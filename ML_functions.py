# ML_functions
"""
Contains different Machine Learning functions, like

"""

import tensorflow as tf
from photonpy import PostProcessMethods, Context
import numpy as np


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
    neighbours : 2xN float array
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
         
    return neighbours_ch1, neighbours_ch2


def vicinity_neighbours_numpy(ch1, ch2, maxDistance):
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
    neighbours : 2xN float array
        An array containing in the first column the localization index in ch1 
        and in the second column the localization index of its neighbours in ch2.

    '''
    neighbours_ch1 = tf.Variable([], tf.int32)
    neighbours_ch2 = tf.Variable([], tf.int32)
    N = ch1.shape[1]

    for i in range(N): 
        distance = ( ch2[0]**2+ ch2[1]**2 ) - ( ch1[0]**2+ ch1[1]**2 ) 
        nn_index = tf.math.argmin(distance)
        nn_index1 = tf.keras.backend.eval(nn_index)
        print(nn_index1)
        
        neighbours_ch1 = tf.concat([ neighbours_ch1, [ nn_index1 ] ], axis = 0)
        neighbours_ch2 = tf.concat([ neighbours_ch2, [int(i)] ], axis = 0)
    return neighbours_ch1, neighbours_ch2
        
        