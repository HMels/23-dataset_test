# output_text.py
"""
Created on Thu Apr 22 14:26:22 2021

@author: Mels
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import copy
import tensorflow as tf


from generate_neighbours import KNN

#%% error handling of batches
def Info_batch(N):
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
    print('\nI: the total system contains', N, ' points. The setup seems to be OK',
          '\nNote that for big N, batches shoud be used.\n')
    time.sleep(2)


#%% Error distribution 
def errorHist(ch1, ch2, ch2m, bin_width = 5, plot_on=True, direct=False):
    '''
    Generates a histogram showing the distribution of distances between coupled points

    Parameters
    ----------
    ch1 : Nx2
        The localizations of channel 1.
    ch2 , ch2m : Nx2
        The localizations of channel 2 and the mapped channel 2. The indexes 
        of should be one-to-one with channel 1
    bin_width : int, optional
        The width of a bin. The default is 20.
    plot_on : bool, optional
        Do we want to plot. The default is True.
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.

    Returns
    -------
    avg1, avg2 : float
        The average distance between the channels

    '''
    if direct:
        dist1 = np.sqrt( np.sum( ( ch1 - ch2 )**2, axis = 1) )
        dist2 = np.sqrt( np.sum( ( ch1 - ch2m )**2, axis = 1) )
        
        avg1 = np.average(dist1)
        avg2 = np.average(dist2)
            
    else:
        idx1 = KNN(ch1, ch2, 1)
        idx2 = KNN(ch1, ch2m, 1)
        
        dist1 = np.sqrt( np.sum( ( ch1 - ch2[idx1,:][:,0,:] )**2, axis = 1) )
        dist2 = np.sqrt( np.sum( ( ch1 - ch2m[idx2,:][:,0,:] )**2, axis = 1) )
        
        avg1 = np.average(dist1)
        avg2 = np.average(dist2)
            
    if plot_on:
        nbins1 = round(( np.max(dist1) - np.min(dist1) ) / bin_width ,0).astype(int)
        nbins2 = round(( np.max(dist2) - np.min(dist2) ) / bin_width ,0).astype(int)
    
        plt.figure()
        plt.title('Distribution of distances between neighbouring Localizations')
        n1 = plt.hist(dist1+.25, label='Original', alpha=.8, edgecolor='red', bins=nbins1)
        n2 = plt.hist(dist2, label='Mapped', alpha=.7, edgecolor='yellow', bins=nbins2)
            
        ymax = np.max([np.max(n1[0]), np.max(n2[0])]) + 5
        plt.vlines(avg1, color='purple', ymin=0, ymax=ymax)
        plt.vlines(avg2, color='green', ymin = 0, ymax=ymax)
            
        plt.ylim([0,ymax])
        plt.xlim(0)
        plt.xlabel('distance [nm]')
        plt.ylabel('# of localizations')
        plt.legend()
        plt.show()
    
    return avg1, avg2


#%% Error distribution over FOV 
def errorFOV(ch1, ch2, ch2m, plot_on=True, direct=False):
    '''
    Generates a FOV distribution of distances between coupled points

    Parameters
    ----------
    ch1 : Nx2
        The localizations of channel 1.
    ch2 , ch2m : Nx2
        The localizations of channel 2 and the mapped channel 2. The indexes 
        of should be one-to-one with channel 1
    bin_width : int, optional
        The width of a bin. The default is 20.
    plot_on : bool, optional
        Do we want to plot. The default is True.
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.

    Returns
    -------
    avg1, avg2 : float
        The average distance between the channels

    '''
    if direct:
        r = np.sqrt(np.sum(ch1**2,1))
        error1 = np.sqrt( np.sum( ( ch1 - ch2 )**2, axis = 1) )
        error2 = np.sqrt( np.sum( ( ch1 - ch2m )**2, axis = 1) )
        
        avg1 = np.average(error1)
        avg2 = np.average(error2)
            
    else:
        r = np.sqrt(np.sum(ch1**2,1))
        idx1 = KNN(ch1, ch2, 1)
        idx2 = KNN(ch1, ch2m, 1)
        error1 = np.sqrt( np.sum( ( ch1 - ch2[idx1,:][:,0,:] )**2, axis = 1) )
        error2 = np.sqrt( np.sum( ( ch1 - ch2m[idx2,:][:,0,:] )**2, axis = 1) )
        
        avg1 = np.average(error1)
        avg2 = np.average(error2)
        
    if plot_on:
        plt.figure()
        plt.title('Distribution of error between neighbouring pairs over radius')
        plt.plot(r, error1, 'b.', alpha=.3, label='Original error')
        plt.plot(r, error2, 'r.', alpha=.4, label='Mapped error')
        plt.xlabel('Distance from center [nm]')
        plt.ylabel('Error [nm]')
        plt.legend()
    
    return avg1, avg2


#%% Plotting the grid
def plot_grid(ch1, ch2, ch2_map, mods, gridsize=50, d_grid=.1, lines_per_CP=1, 
              locs_markersize=10, CP_markersize=8, grid_markersize=3, grid_opacity=1): 
    '''
    Plots the grid and the shape of the grid in between the Control Points

    Parameters
    ----------
    ch1 , ch2 , ch2_map : Nx2 tf.float32 tensor
        The tensor containing the localizations.
    mods : Models() Class
        The Model which has been trained on the dataset.
    gridsize : float, optional
        The size of the grid used in mods. The default is 50.
    d_grid : float, optional
        The precission of the grid we want to plot in between the
        Control Points. The default is .1.
    locs_markersize : float, optional
        The size of the markers of the localizations. The default is 10.
    CP_markersize : float, optional
        The size of the markers of the Controlpoints. The default is 8.
    grid_markersize : float, optional
        The size of the markers of the grid. The default is 3.
    grid_opacity : float, optional
        The opacity of the grid. The default is 1.
    lines_per_CP : int, optional
        The number of lines we want to plot in between the grids. 
        Works best if even. The default is 1.

    Returns
    -------
    None.

    '''
    # system parameters
    x1_min = tf.reduce_min(tf.floor(ch2[:,0]/gridsize))
    x1_max = tf.reduce_max(tf.floor(ch2[:,0]/gridsize))
    x2_min = tf.reduce_min(tf.floor(ch2[:,1]/gridsize))
    x2_max = tf.reduce_max(tf.floor(ch2[:,1]/gridsize))
    
    # Creating the horizontal grid
    grid_tf = []
    x1_grid = tf.range(x1_min, x1_max+1, d_grid)
    x2_grid = x2_min * tf.ones(x1_grid.shape[0], dtype=tf.float32)
    while x2_grid[0] < x2_max+.99:
        grid_tf.append(tf.concat((x1_grid[:,None], x2_grid[:,None]), axis=1))
        x2_grid +=  np.round(1/lines_per_CP,2)
    
    # Creating the vertical grid
    x2_grid = tf.range(x2_min, x2_max+1, d_grid)
    x1_grid = x1_min * tf.ones(x2_grid.shape[0], dtype=tf.float32)
    while x1_grid[0] < x1_max+.99:
        grid_tf.append(np.concatenate((x1_grid[:,None], x2_grid[:,None]), axis=1))
        x1_grid += np.round(1/lines_per_CP,2)
        
    # Adding to get the original grid 
    grid_tf = tf.concat(grid_tf, axis=0)
    CP_idx = tf.cast(tf.stack(
            [( grid_tf[:,0]-tf.reduce_min(tf.floor(grid_tf[:,0]))+1)//1 , 
             ( grid_tf[:,1]-tf.reduce_min(tf.floor(grid_tf[:,1]))+1)//1 ], 
            axis=1), dtype=tf.int32)
    
    # transforming the grid
    mods_temp = copy(mods.model)
    mods_temp.reset_CP(CP_idx)
    grid_tf = mods_temp.transform_vec(grid_tf)
    
    # plotting the localizations
    plt.figure()
    plt.plot(ch2_map[:,0],ch2_map[:,1], color='red', marker='.', linestyle='',
             markersize=locs_markersize, label='Mapped CH2')
    plt.plot(ch2[:,0],ch2[:,1], color='orange', marker='.', linestyle='', 
             alpha=.7, markersize=locs_markersize-2, label='Original CH2')
    plt.plot(ch1[:,0],ch1[:,1], color='blue', marker='.', linestyle='', 
             markersize=locs_markersize, label='Original CH1')
    
    # plotting the grid
    plt.plot(grid_tf[:,0]*gridsize,grid_tf[:,1]*gridsize, 'g.',
             markersize=grid_markersize, alpha=grid_opacity)
    plt.plot( mods.model.CP_locs[:,:,0]*gridsize,  mods.model.CP_locs[:,:,1]*gridsize, 
             'b+', markersize=CP_markersize)
    plt.legend()
