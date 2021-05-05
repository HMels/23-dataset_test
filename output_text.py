# output_text.py
"""
Created on Thu Apr 22 14:26:22 2021

@author: Mels
"""
import numpy as np
import tensorflow as tf
import time

import setup_image

#%% error handling of batches
def Info_batch(N, num_batches, batch_size, Batch_on=True):
    '''
    Handles the error if batches are sufficient large

    Parameters
    ----------
    N : int
        total amount of points.
    num_batches : 2 int array
        containing the amount of [x1,x2] batches.
    batch_size : int
        max number of points per batch.
    Batch_on : bool, optional
        Are batches used. The default is True.

    Returns
    -------
    None.

    '''
    if Batch_on:
        perc = np.prod(num_batches)*batch_size / N * 100
        if perc < 100:
            print('\nI: in the current setup (',num_batches[0],'x',num_batches[1],'batches, ',
                  batch_size,' points per batch and ',N, ' points total)',
                  ' an estimate of ', round(perc,2),'\% of points are used to calculate the minimum entropy.\n')
            input("Press Enter to continue...")
        else: 
            print('\nI: in the current setup (',num_batches[0],'x',num_batches[1],'batches, ',
                  batch_size,' points per batch and ',N, ' points total)',
                  ' an estimate of ', round(perc,2),'\% of points are used to calculate the minimum entropy. \nThe setup seems to be OK and the amount of Batches is sufficient.\n')
            time.sleep(2)
    else: 
        print('\nI: the total system contains', N, ' points. The setup seems to be OK',
              '\nNote that for big N, batches shoud be used.\n')
        time.sleep(2)


#%% overlap
def overlap(ch1, ch2):
    '''
    Calculates the overlap of the channels

    Parameters
    ----------
    ch1 , ch2 : NxM np.array
        Array containing the information of the image.

    Returns
    -------
    float
        The overlap metric.

    '''
    overlap = ch1*ch2
    return np.sum(np.sum(overlap))
    

def avg_dist(ch1, ch2):
    dist = np.sqrt( np.sum( ( ch1 - ch2 )**2, axis = 1) )
    return np.average(dist)
    

#%% Polynomial translation
def polynomial_translation(locs, M1, M2):
    m = M1.shape[0]
    y = np.zeros(locs.shape)[None]
    
    for i in range(m):
        for j in range(m):
            y1=  np.array([
                M1[i,j] * (locs[:,0]**i) * ( locs[:,1]**j),
                M2[i,j] * (locs[:,0]**i) * ( locs[:,1]**j)
                ]).transpose()[None]
            y = np.concatenate([y, y1 ], axis = 0) 
    return np.sum(y, axis = 0)
