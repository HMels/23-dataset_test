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
        ch1, ch2, maxDistance=maxDistance)
    
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
        