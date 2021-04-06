# distributions.py
'''
This Module contains the next functions
- gauss_2d()
- circle_2d_normal
- circle_2d_uniform()
'''

import numpy as np
from numpy import random as rnd


def gauss_2d(mu, sigma, N):
    '''
generates a 2d gaussian function 
the intensity (or z direction) is uniformly distributed
output is the [x1, x2] locations of the spots
    '''
    x1 = rnd.normal(mu[0], sigma[0], N)
    x2 = rnd.normal(mu[1], sigma[1], N)
    return np.array([x1, x2])


def circle_2d_normal(mu, sigma, N, width):
    '''
generates a circular distribution
the radius is normal distributed, the angle and intensity uniformly 
output is the [x1, x2] locations of the spots
   '''
    alpha = 2 * np.pi * rnd.random(N)
    rx1 = (np.abs(rnd.normal(sigma[0], width, N))) 
    rx2 = (np.abs(rnd.normal(sigma[1], width, N)))
    
    x1 = rx1 * np.sin(alpha) + mu[0]
    x2 = rx2 * np.cos(alpha) + mu[1]
    return np.array([x1, x2])


def circle_2d_uniform(mu, sigma, N, width = None):
    '''
generates a circular distribution
the radius is normal distributed, the angle and intensity uniformly
output is the [x1, x2] locations of the spots
    '''
    alpha = 2 * np.pi * rnd.random(N)
    rx1 = sigma[0] * np.sqrt(rnd.random(N)) 
    rx2 = sigma[1] * np.sqrt(rnd.random(N))
    
    x1 = rx1 * np.sin(alpha) + mu[0]
    x2 = rx2 * np.cos(alpha) + mu[1]
    return np.array([x1, x2])
