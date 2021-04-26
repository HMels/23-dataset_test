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
def simple_translation(locs, shift, rotation):
    return shift_translation( rotation_translation( locs, rotation), shift)


def complex_translation(locs, shift, rotation, shear, scaling):
    return shift_translation( rotation_translation(
        shear_translation(scaling_translation(locs, scaling), shear),
        rotation), shift)


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
    return tf.reduce_sum(y, axis = 0)


#%% The translation functions
def shift_translation(locs, shift):
    '''
    shifts the localizations

    Parameters
    ----------
    locs: Nx2 matrix float
        The actual locations of the localizations.
    shift : 2 float array
        The shift of the image in nm.

    Returns
    -------
    locs: Nx2 matrix float
        The actual locations of the localizations.

    '''
    locs[:,0] += shift[0]
    locs[:,1] += shift[1]
    return locs


def rotation_translation(locs, rotation):
    '''
    rotates the localizations

    Parameters
    ----------
    locs: Nx2 matrix float
        The actual locations of the localizations.
    angle : float
        The angle of rotation in degrees.

    Returns
    -------
    locs: Nx2 matrix float
        The actual locations of the localizations.

    '''
    cos = np.cos(rotation * 0.0175/100) 
    sin = np.sin(rotation * 0.0175/100)
   
    locs = np.array([
         (cos * locs[:,0] - sin * locs[:,1]) ,
         (sin * locs[:,0] + cos * locs[:,1]) 
        ]).transpose()
    return locs


def shear_translation(locs, shear):
    '''
    Deforms the localizations with a shear translation

    Parameters
    ----------
    locs: Nx2 matrix float
        The actual locations of the localizations.
    angle : float 2 array
        The [x1,x2] shear translation

    Returns
    -------
    locs: Nx2 matrix float
        The actual locations of the localizations.

    '''
    locs = np.array([
        locs[:,0] + shear[0]*locs[:,1] ,
        shear[1]*locs[:,0] + locs[:,1] 
        ]).transpose()
    return locs


def scaling_translation(locs, scaling):
    '''
    deforms the localizations with a scaling

    Parameters
    ----------
    locs: Nx2 matrix float
        The actual locations of the localizations.
    scaling : float 2 array
        The [x1,x2] scaling.

    Returns
    -------
    locs: Nx2 matrix float
        The actual locations of the localizations.

    '''
    locs = np.array([
        scaling[0] * locs[:,0] ,
        scaling[1] * locs[:,1]
        ]).transpose()
    return locs