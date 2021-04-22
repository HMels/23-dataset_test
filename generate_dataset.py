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


def generate_localizations(cluster):
    '''
    Generates the localizations via a certain distribution 

    Parameters
    ----------
    cluster : Cluster() array
        Class containing the data of the clusters needed to be generated.

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    localizations = np.empty([0,2], dtype = float)
    for clus in cluster: 
        new_cluster = distributions.gauss_2d( clus.loc()  , clus.std(), clus.num() )
        localizations = np.append( localizations, new_cluster, 0)
    return localizations 


def localization_error(localizations, error = 10):
    '''
    Generates a Gaussian localization error over the localizations 

    Parameters
    ----------
    localizations: 2xN matrix float
        The actual locations of the localizations.
    error : float, optional
        The localization error in nm. The default is 10

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    N = len(localizations[:,0])
    localizations[:,0] += rnd.normal(0, error, N)
    localizations[:,1] += rnd.normal(0, error, N)
    return localizations
        

def generate_noise(localizations, img, Noise):
    '''
    Parameters
    ----------
    localizations: 2xN matrix float
        The actual locations of the localizations.
    img: 2x2 array 
        containing the border values of the system
    Noise: float
        The percentage of Noice added to the system

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    N_Noise = int(Noise * localizations.shape[0])
    
    img_size = img[1,:] - img[0,:] 

    Noise_loc = np.array([
        img_size[0] * ( rnd.rand( N_Noise ) -0.5) ,
        img_size[1] * ( rnd.rand( N_Noise ) -0.5)
        ])
    
    return np.append(localizations, np.squeeze( Noise_loc.transpose() ), 0)