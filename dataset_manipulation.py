# dataset_manipulation.py
'''
This script contains the functions needed to perform manipulations and deformations 
on the localizations.

This Module contains the next functions
- shift()
- rotation()
- shear()
- scaling
'''

import numpy as np
import tensorflow as tf

#%% Main translation
def simple_translation(localizations, shift, rotation):
    return shift_translation( rotation_translation( localizations, rotation), shift)


def complex_translation(localizations, shift, rotation, shear, scaling):
    return shift_translation( rotation_translation(
        shear_translation(scaling_translation(localizations, scaling), shear),
        rotation), shift)


#%% Polynomial translation
def polynomial_translation(localizations, M1, M2):
    m = M1.shape[0]
    y = np.zeros(localizations.shape)[None]
    
    for i in range(m):
        for j in range(m):
            y1=  np.array([
                M1[i,j] * (localizations[:,0]**i) * ( localizations[:,1]**j),
                M2[i,j] * (localizations[:,0]**i) * ( localizations[:,1]**j)
                ]).transpose()[None]
            y = np.concatenate([y, y1 ], axis = 0) 
    return tf.reduce_sum(y, axis = 0)


#%% The translation functions
def shift_translation(localizations, shift):
    '''
    shifts the localizations

    Parameters
    ----------
    localizations: Nx2 matrix float
        The actual locations of the localizations.
    shift : 2 float array
        The shift of the image in nm.

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    localizations[:,0] += shift[0]
    localizations[:,1] += shift[1]
    return localizations


def rotation_translation(localizations, rotation):
    '''
    rotates the localizations

    Parameters
    ----------
    localizations: Nx2 matrix float
        The actual locations of the localizations.
    angle : float
        The angle of rotation in radians.

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    cos = np.cos(rotation) 
    sin = np.sin(rotation)
   
    localizations = np.array([
         (cos * localizations[:,0] - sin * localizations[:,1]) ,
         (sin * localizations[:,0] + cos * localizations[:,1]) 
        ]).transpose()
    return localizations


def shear_translation(localizations, shear):
    '''
    Deforms the localizations with a shear translation

    Parameters
    ----------
    localizations: Nx2 matrix float
        The actual locations of the localizations.
    angle : float 2 array
        The [x1,x2] shear translation

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    localizations = np.array([
        localizations[:,0] + shear[0]*localizations[:,1] ,
        shear[1]*localizations[:,0] + localizations[:,1] 
        ]).transpose()
    return localizations


def scaling_translation(localizations, scaling):
    '''
    deforms the localizations with a scaling

    Parameters
    ----------
    localizations: Nx2 matrix float
        The actual locations of the localizations.
    scaling : float 2 array
        The [x1,x2] scaling.

    Returns
    -------
    localizations: Nx2 matrix float
        The actual locations of the localizations.

    '''
    localizations = np.array([
        scaling[0] * localizations[:,0] ,
        scaling[1] * localizations[:,1]
        ]).transpose()
    return localizations