# dataset_manipulation.py
'''
This script contains the functions needed to perform manipulations and deformations 
on the localizations.

This Module contains the next functions
- shift()
- rotation()
'''

import numpy as np


def shift(localizations, shift):
    '''
    shifts the localizations by shift

    Parameters
    ----------
    localizations: Nx2 matrix float
        The actual locations of the localizations.
    shift : 2 float array
        The shift of the image in zoom parameters.

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    localizations[:,0] += shift[0]
    localizations[:,1] += shift[1]
    return localizations


def rotation(localizations, angle, img_param):
    '''
    displace the localizations to have the axis of rotation in the exact middle, 
    then it rotates the localizations by an angle, and then it displaces the 
    localizations back 

    Parameters
    ----------
    localizations: Nx2 matrix float
        The actual locations of the localizations.
    angle : float
        The angle of rotation in degrees.
    img_param : Image()
        Class containing the data of the image.

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    cos = np.cos(angle) 
    sin = np.sin(angle)
   
    mid= np.array([0,0])# img_param.img_size_zoom()/2  
    localizations[:,0] = localizations[:,0] - mid[0]
    localizations[:,1] = localizations[:,1] - mid[1]
    
    localizations = np.array([
         (cos * localizations[:,0] - sin * localizations[:,1]) ,
         (sin * localizations[:,0] + cos * localizations[:,1]) 
        ]).transpose()
    
    localizations[:,0] = localizations[:,0] + mid[0]
    localizations[:,1] = localizations[:,1] + mid[1]
    return localizations