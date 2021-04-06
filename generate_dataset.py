# generate_dataset.py
'''
This module  contains the next functions
- generate_localizations()
- localization_error()
- generate_channel()
'''

import numpy as np 
import numpy.random as rnd

import distributions
import functions


def generate_localizations(cluster, img_param):
    '''
generates the localizations by iterating over the different clusters, 
generate via a gaussian distributino and adding them together in localizations
Localizations consist of 3 colums. The [x1, x2] position of the localization
and the third column has the intensity of named localization
    '''
    pix = img_param.pix_size_zoom()
    localizations = np.array([[], []])
    for clus in cluster: 
        new_cluster = distributions.gauss_2d( clus.loc() * pix  , clus.std() * pix, clus.num() )
        localizations = np.append( localizations, new_cluster, 1)
    return localizations 


def localization_error(localizations, img_param, error = 0.1):
    '''
generates a gaussian localization error (in pixels)
output is the localization with the error included
    '''
    error = error * img_param.pix_size_zoom()
    N = len(localizations[0,:])
    localizations[0,:] += rnd.normal(0, error, N)
    localizations[1,:] += rnd.normal(0, error, N)
    return localizations
        

def generate_noise(img_param, localizations):
    '''
    

    Parameters
    ----------
    img_param
    localizations : float array
        containing all localizations of a channel

    Returns
    -------
    localizations including noise

    '''
    img_size = img_param.img_size_zoom()

    Noise_loc = np.array([
        [ img_size[0] * rnd.rand( img_param.Noise ) ],
        [ img_size[1] * rnd.rand( img_param.Noise ) ]
        ])
    localizations = np.append(localizations, np.squeeze(Noise_loc), 1)
    
    return localizations


def generate_channel(img_param, localizations):
    '''
takes the generated locations and fills them into the channel matrix
output is an NxN matrix with intensities of localizations and Noise. 
Plot via plt.imshow(channel)
    '''
    img_size = img_param.img_size_zoom()
    
    channel = np.zeros([ int(img_size[0]), int(img_size[1]) ], dtype = int)
    for i in range(len(localizations[0,:])):
        loc = localizations[:,i]
        if functions.isin_domain(loc, img_param):
            channel[int(loc[0])-1, int(loc[1])-1] = 1
    return channel