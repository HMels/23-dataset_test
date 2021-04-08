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
    Generates the localizations via a certain distribution 

    Parameters
    ----------
    cluster : Cluster() array
        Class containing the data of the clusters needed to be generated.
    img_param : Image()
        Class containing the data of the image.

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    pix = img_param.pix_size_zoom()
    localizations = np.empty([0,2], dtype = float)
    for clus in cluster: 
        new_cluster = distributions.gauss_2d( clus.loc() * pix  , clus.std() * pix, clus.num() )
        localizations = np.append( localizations, new_cluster, 0)
    return localizations 


def localization_error(localizations, img_param, error = 0.1):
    '''
    Generates a Gaussian localization error over the localizations 

    Parameters
    ----------
    localizations: Nx2 matrix float
        The actual locations of the localizations.
    img_param : Image()
        Class containing the data of the image.
    error : float, optional
        The localization error in pixels. The default is 0.1.

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    error = error * img_param.pix_size_zoom()
    N = len(localizations[:,0])
    localizations[:,0] += rnd.normal(0, error, N)
    localizations[:,1] += rnd.normal(0, error, N)
    return localizations
        

def generate_noise(img_param, localizations):
    '''
    

    Parameters
    ----------
    img_param : Image()
        Class containing the data of the image.
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    img_size = img_param.img_size_zoom()

    Noise_loc = np.array([
        img_size[0] * rnd.rand( img_param.Noise ) ,
        img_size[1] * rnd.rand( img_param.Noise ) 
        ])
    localizations = np.append(localizations, np.squeeze( Noise_loc.transpose() ), 0)
    
    return localizations


def generate_channel(img_param, localizations):
    '''
    Takes the localizations and puts them in a matrix

    Parameters
    ----------
    img_param : Image()
        Class containing the data of the image.
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    Returns
    -------
    channel : matrix
        Contains an image of the localizations.

    '''
    img_size = img_param.img_size_zoom()
        
    localizations[:,0] = localizations[:,0] + img_param.img_size_zoom()[0]/2 
    localizations[:,1] = localizations[:,0] + img_param.img_size_zoom()[1]/2 
    
    channel = np.zeros([ int(img_size[0]), int(img_size[1]) ], dtype = int)
    for i in range(len(localizations[:,0])):
        loc = localizations[i,:]
        if functions.isin_domain(loc, img_param):
            channel[int(loc[0])-1, int(loc[1])-1] = 1
    return channel