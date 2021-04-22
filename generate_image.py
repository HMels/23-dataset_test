# generate_image.py
"""
Created on Mon Apr 19 14:09:07 2021

@author: Mels
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_channel(locs1, locs2, locs3, precision = 10, pix_size = 100):
    '''
    Generates the image of all three figures

    Parameters
    ----------
    locs1, locs2, locs3 : Nx2 float array
        The localizations.
    precision : float, optional
        The amount of precision in nm. The default is 10
    pix_size : int, optional
        The size of the pixels in nm. The default is 100.

    Returns
    -------
    None.

    '''
    zoom = pix_size / precision

    # normalizing system
    locs1 = locs1 * zoom
    locs2 = locs2 * zoom
    locs3 = locs3 * zoom
    
    # calculate bounds of the system
    bounds = np.empty([2,2], dtype = float) 
    bounds[0,0] = np.min([ np.min(locs1[:,0]), np.min(locs2[:,0]), np.min(locs3[:,0]) ])
    bounds[0,1] = np.max([ np.max(locs1[:,0]), np.max(locs2[:,0]), np.max(locs3[:,0]) ])
    bounds[1,0] = np.min([ np.min(locs1[:,1]), np.min(locs2[:,1]), np.min(locs3[:,1]) ])
    bounds[1,1] = np.max([ np.max(locs1[:,1]), np.max(locs2[:,1]), np.max(locs3[:,1]) ])
    
    # generating the matrices to be plotted
    channel1 = generate_channel(locs1, bounds)
    channel2 = generate_channel(locs2, bounds)
    channel3 = generate_channel(locs3, bounds)
    
    axis = np.array([ bounds[1,:], bounds[0,:]]) * precision
    axis = np.reshape(axis, [1,4])[0]
    
    # plotting all channels
    plt.figure()
    plt.subplot(131)
    plt.imshow(channel1, extent = axis)
    plt.title('original channel 1')

    plt.subplot(132)
    plt.imshow(channel2, extent = axis)
    plt.title('original channel 2')

    plt.subplot(133)
    plt.imshow(channel3, extent = axis)
    plt.title('channel 2 mapped')
    
    
    # comparisson between original 1 and mapped 2 
    plt.figure()
    plt.subplot(121)
    plt.imshow(channel1, extent = axis)
    plt.title('original channel 1')
    
    plt.subplot(122)
    plt.imshow(channel3, extent = axis)
    plt.title('channel 2 mapped')



def generate_channel(locs , bounds):
    '''
    Takes the localizations and puts them in a matrix
    Parameters
    ----------
    img_param : Image()
        Class containing the data of the image.
    localizations : Nx2 matrix float
        The actual locations of the localizations.
    bounds : 2x2 matrix 
        containing the bounds of all three systems
        
    Returns
    -------
    channel : matrix
        Contains an image of the localizations.
    '''
    size_img = np.round( (bounds[:,1] - bounds[:,0]) , 0).astype('int')

    channel = np.zeros([size_img[0], size_img[1]], dtype = int)
    for i in range(locs.shape[0]):
        loc = np.round(locs[i,:],0).astype('int')
        if isin_domain(loc, bounds):
            loc -= np.round(bounds[:,0],0).astype('int') # place the zero point on the left
            channel[loc[0]-1, loc[1]-1] = 1
    return channel


def isin_domain(pos, img):
    '''
    checks if pos is within img domain

    Parameters
    ----------
    pos : 2 float array
        Position of a certain point.
    img : 2x2 float array
        The bounds of a certain image.

    Returns
    -------
    Bool
        True if pos is in img, False if not.

    '''
    return ( pos[0] > img[0,0] and pos[1] > img[1,0] and 
            pos[0] < img[0,1] and pos[1] < img[1,1] )

