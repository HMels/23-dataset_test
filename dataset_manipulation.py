# dataset_manipulation.py
'''
This Module contains the next functions
- shift()
- rotation()
'''

import numpy as np


def shift(localizations, shift):
    '''
shifts the localization vector
    '''
    localizations[0,:] += shift[0]
    localizations[1,:] += shift[1]
    return localizations


def rotation(localizations, angle, img_param):
    '''
displace the localizations to have the axis of rotation in the exact middle, 
then it rotates the localizations by an angle, and then it displaces the 
localizations back 
    '''
    cos = np.cos(angle) 
    sin = np.sin(angle)
    
    mid= img_param.img_size_zoom()/2  
    localizations[0,:] = localizations[0,:] - mid[0]
    localizations[1,:] = localizations[1,:] - mid[1]
    
    localizations[:2,:] = np.array([
         np.transpose(cos * localizations[0,:] - sin * localizations[1,:]) ,
         np.transpose(sin * localizations[0,:] + cos * localizations[1,:]) 
        ])
    
    localizations[0,:] = localizations[0,:] + mid[0]
    localizations[1,:] = localizations[1,:] + mid[1]
    return localizations