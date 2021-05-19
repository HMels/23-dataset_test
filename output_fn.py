# output_text.py
"""
Created on Thu Apr 22 14:26:22 2021

@author: Mels
"""
import numpy as np
import time
import matplotlib.pyplot as plt

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
def errorHist_direct(ch1, ch2, ch2m, bin_width = 5, plot_on=True):
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

    Returns
    -------
    avg1, avg2 : float
        The average distance between the channels

    '''
    dist1 = np.sqrt( np.sum( ( ch1 - ch2 )**2, axis = 1) )
    dist2 = np.sqrt( np.sum( ( ch1 - ch2m )**2, axis = 1) )
    
    avg1 = np.average(dist1)
    avg2 = np.average(dist2)
    
    if plot_on:
        nbins1 = round(( np.max(dist1) - np.min(dist1) ) / bin_width ,0).astype(int)
        nbins2 = round(( np.max(dist2) - np.min(dist2) ) / bin_width ,0).astype(int)
        
        plt.figure()
        plt.title('Distribution of distances between coupled Localizations')
        if nbins1>0:
            n1 = plt.hist(dist1+.25, label='Original', alpha=.8, edgecolor='red', bins=nbins1)
            ymax1 = np.max(n1[0])
        else: ymax1=10
        if nbins2>0:
            n2 = plt.hist(dist2, label='Mapped', alpha=0.7, edgecolor='yellow', bins=nbins2)
            ymax2 = np.max(n2[0])
        else: ymax2=10
            
        ymax = np.max([ymax1, ymax2]) + 5
        plt.vlines(avg1, color='purple', ymin=0, ymax=ymax, label='Original Average Distance')
        plt.vlines(avg2, color='green', ymin = 0, ymax=ymax, label='Mapped Average Distance')
        
        plt.ylim([0,ymax])
        plt.xlim(0)
        plt.xlabel('distance [nm]')
        plt.ylabel('# of localizations')
        plt.legend()
        plt.show()
    
    return avg1, avg2


def errorHist_neighbours(ch1, ch2, ch2m, bin_width = 5, plot_on=True):
    '''
    Generates a histogram showing the distribution of distances between Neighbours

    Parameters
    ----------
    ch1 : Nx2
        The localizations of channel 1.
    ch2 , ch2m : Nx2
        The localizations of channel 2 and the mapped channel 2.
    bin_width : int, optional
        The width of a bin. The default is 20.
    plot_on : bool, optional
        Do we want to plot. The default is True.

    Returns
    -------
    avg1, avg2 : float
        The average distance between the channels

    '''
    
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
        n2 = plt.hist(dist2, label='Mapped', alpha=0.7, edgecolor='yellow', bins=nbins2)
        
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
def errorFOV_direct(ch1, ch2, ch2m, plot_on=True):
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

    Returns
    -------
    avg1, avg2 : float
        The average distance between the channels

    '''
    r = np.sqrt(np.sum(ch1**2,1))
    error1 = np.sqrt( np.sum( ( ch1 - ch2 )**2, axis = 1) )
    error2 = np.sqrt( np.sum( ( ch1 - ch2m )**2, axis = 1) )
    
    avg1 = np.average(error1)
    avg2 = np.average(error2)
    
    if plot_on:
        plt.figure()
        plt.title('Distribution of error between coupled pairs over radius')
        plt.plot(r, error1, 'ro', label='Original error')
        plt.plot(r, error2, 'bo', label='Mapped error')
        plt.xlabel('Distance from center [nm]')
        plt.ylabel('Error [nm]')
        plt.legend()
    
    return avg1, avg2


def errorFOV_neighbours(ch1, ch2, ch2m, plot_on=True):
    '''
    Generates a FOV distribution of distances between neighbouring points

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

    Returns
    -------
    avg1, avg2 : float
        The average distance between the channels

    '''
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
        plt.plot(r, error1, 'ro', label='Original error')
        plt.plot(r, error2, 'bo', label='Mapped error')
        plt.xlabel('Distance from center [nm]')
        plt.ylabel('Error [nm]')
        plt.legend()
    
    return avg1, avg2


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
