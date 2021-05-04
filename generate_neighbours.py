# generate_neighbours.py
"""
Created on Thu Apr 29 14:05:40 2021

@author: Mels
"""
import numpy as np
from photonpy import PostProcessMethods, Context
import time


def find_bright_neighbours(locs_A, locs_B, threshold = None, maxDistance = 50):
    '''
    generates a list with arrays containing the neighbours via find_channel_neighbours
    It then deletes all none bright spots.  Also used to make sure output matrix has
    uniform size

    Parameters
    ----------
    locs_A, locs_B : 2xN float numpy array
        The locations of the localizations.
    threshold : int, optional
        The threshold of neighbouring locs needed to not be filtered. The default is None,
        which means the program will use a threshold of average + std
    maxDistance : float/int, optional
        The vicinity in which we search for neighbours. The default is 50.
        
    Returns
    -------
    idxlist_new : list
        list filtered on bright spots, with size [2 x threshold]
        containing per indice of ch1 the neighbours in ch2.
    '''
    idxlist = find_channel_neighbours(locs_A, locs_B, maxDistance)

    if threshold == None: # threshold = avg + std
        num = []
        for idx in idxlist:
            if idx != []:
                num.append(idx.shape[1])
        threshold = np.round(np.average(num) + np.std(num),0).astype('int')
    
    print('Filtering brightest spots...')
    idx1list = []
    idx2list = []
    for idx in idxlist:
        if idx != []:
            if idx.shape[1] > threshold:
                # we want to have a max of threshold in our array
                idx1list.append(idx[0,
                                    np.random.choice(idx.shape[1], threshold)
                                    ]) 
                idx2list.append(idx[1,
                                    np.random.choice(idx.shape[1], threshold)
                                    ]) 
    
    if idx1list == []:
        print('\nError: No neighbours generated. Might be related to Threshold!')
        time.sleep(5)
        
    neighbours_A = generate_neighbour_matrix(idx1list, locs_A)
    neighbours_B = generate_neighbour_matrix(idx2list, locs_B)
    return neighbours_A, neighbours_B


def find_channel_neighbours(locs_A, locs_B, maxDistance = 50):
    '''
    generates a list with arrays containing the neighbours

    Parameters
    ----------
    locs_A, locs_B : 2xN float numpy array
        The locations of the localizations.
    maxDistance : float/int, optional
        The vicinity in which we search for neighbours. The default is 50.

    Returns
    -------
    idxlist : list
        List containing per indice of ch1 the neighbours in ch2.

    '''
    print('Finding neighbours...')
    
    with Context() as ctx:
        counts,indices = PostProcessMethods(ctx).FindNeighbors(locs_A, locs_B, maxDistance)
    
    idxlist = []
    pos = 0
    i = 0
    for count in counts:
        idxlist.append( np.stack([
            i * np.ones([count], dtype=int),
            indices[pos:pos+count] 
            ]) )
        pos += count
        i += 1
            
    return idxlist


def generate_neighbour_matrix(idxlist, locs):
    NN = []
    for nn in idxlist:
        NN.append( locs[nn,:] )
    return np.array(NN)
    