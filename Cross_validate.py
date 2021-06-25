# Cross_reference.py

import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import Model
import OutputModules.output_fn as output_fn


#%% cross_validate
def cross_validate(locs_A, locs_B, gridsize, coupled=False, plot_hist=True, plot_FOV=False):
    '''
    Cross validation

    Parameters
    ----------
    locs_A , locs_B : array
        Array containing the localizations of both channels.
    gridsize : int
        The size of the spline grids.
    coupled : bool, optional
        True if the dataset is coupled. The default is False
    plot_hist : bool, optional
        Do we want to plot the distribution of the error. The default is True.
    plot_FOV : bool, optional
        Do we want to plot the error over the FOV. The default is False.

    Returns
    -------
    avg_trained  , avg_crossref : float
        The average error of the trained and cross validated dataset

    '''
    print('Cross Validating...')
    
    ch1_batch1, ch2_batch1, ch1_batch2, ch2_batch2, ch1, ch2, sys_param = (
        split_batches(locs_A,locs_B, direct=coupled) )
    
    
    ## Train model on first batch
    mods1, ch2_trained = Model.run_model(ch1_batch1, ch2_batch1, coupled=coupled,
                                        N_it=[400, 200], learning_rate=[1,1e-2], 
                                        gridsize=gridsize, plot_grid=False, sys_param=sys_param)
    
    ch2_mapped = Model.transform_vec(mods1, ch2_batch2, gridsize, sys_param=sys_param)
    
    
    ## Plot Hist
    nbins = 30                                          # Number of bins
    _, avg_trained, fig1, (ax11, ax12) = output_fn.errorHist(
        ch1_batch1,  ch2_batch1, ch2_trained, nbins=nbins, direct=coupled, plot_on=plot_hist
        )
    _, avg_crossref, fig2, (ax21, ax22) = output_fn.errorHist(
        ch1_batch2,  ch2_batch2, ch2_mapped, nbins=nbins, direct=coupled, plot_on=plot_hist
        )
    if plot_hist: ## Changing figures
        fig1.suptitle('Error of trained dataset')
        fig2.suptitle('Cross Reference')
        
    ## Plot FOV
    if plot_FOV:
        _, _, fig1, (ax11, ax12) = output_fn.errorFOV(ch1_batch1,  ch2_batch1, ch2_trained, direct=coupled)
        _, _, fig2, (ax21, ax22) = output_fn.errorFOV(ch1_batch2,  ch2_batch2, ch2_mapped, direct=coupled)
    
    return avg_trained, avg_crossref


#%% cross_validate_fns
def split_batches(locs_A,locs_B, direct=False):
    '''
    

    Parameters
    ----------
    locs_A , locs_B : np.array
        The locations of the localizations.
    direct : bool, optional
        True if the datet is coupled. The default is False.

    Returns
    -------
    ch1_batch1 , ch2_batch1 , ch1_batch2 , ch2_batch2 : tf Tensor
        The original dataset split into random halves.
    ch1 , ch2 : tf Tensor
        The original datasets.
    sys_params : list, optional
        List containing the size of the system. The optional is None,
        which means it will be calculated by hand

    '''
    
    Nmax = locs_A.shape[0]
    if direct: # if ds is coupled
        if Nmax != locs_B.shape[0]:
            print('Error: Coupled arrays should have same size')
            
        N1 = int(Nmax/2)        
        N1_idx = split_array_idx(N1, Nmax)
        N2_idx = np.linspace(0,Nmax-1,Nmax).astype('int')
        mask = np.ones(Nmax, dtype=bool)
        mask[N1_idx]=False
        N2_idx=N2_idx[mask]
        
        locs_A1 = locs_A[N1_idx,:]
        locs_A2 = locs_A[N2_idx,:]
        locs_B1 = locs_B[N1_idx,:]
        locs_B2 = locs_B[N2_idx,:]
    else:
        N1A = int(Nmax/2)
        N1B = int(Nmax/2)
        
        N1A_idx = split_array_idx(N1A, Nmax)
        N1B_idx = split_array_idx(N1B, Nmax)
        N2A_idx = np.linspace(0,Nmax-1,Nmax).astype('int')
        N2B_idx = np.linspace(0,Nmax-1,Nmax).astype('int')
        
        maskA = np.ones(Nmax, dtype=bool)
        maskA[N1A_idx]=False
        N2A_idx=N2A_idx[maskA]
        
        maskB = np.ones(Nmax, dtype=bool)
        maskB[N1B_idx]=False
        N2B_idx=N2B_idx[maskB]
        
        locs_A1 = locs_A[N1A_idx,:]
        locs_A2 = locs_A[N2A_idx,:]
        locs_B1 = locs_B[N1B_idx,:]
        locs_B2 = locs_B[N2B_idx,:]
    
    
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)
    ch1_batch1 = tf.Variable( locs_A1, dtype = tf.float32)
    ch2_batch1 = tf.Variable( locs_B1, dtype = tf.float32)
    ch1_batch2 = tf.Variable( locs_A2, dtype = tf.float32)
    ch2_batch2 = tf.Variable( locs_B2, dtype = tf.float32)
    
    sys_param = tf.Variable([ [ tf.reduce_min(ch2[:,0]), tf.reduce_min(ch2[:,1]) ],
                              [ tf.reduce_max(ch2[:,0]), tf.reduce_max(ch2[:,1]) ] ])
    
    return ch1_batch1, ch2_batch1, ch1_batch2, ch2_batch2, ch1, ch2, sys_param
    
    
def split_array_idx(N, Nmax):
    '''
    generates an array of size N containing the random indices  of array with size Nmax

    Parameters
    ----------
    N : int
        Size of the array containing the indices.
    Nmax : int
        Size of the array of which we want to calculate the indices.

    Returns
    -------
    np.array
        The array containing the indices.

    '''
    if N>Nmax:
        print('Error: split_array_idx() can only split array into smaller N')
        
    N_idx = np.unique(rnd.randint(0,Nmax,N)).tolist()
    while len(N_idx) < N:
        N_new = rnd.randint(0,Nmax,1).tolist()[0]
        if N_new not in N_idx:
            N_idx.append(N_new)
    return np.sort(np.array(N_idx))

