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
    Generates a 2D gaussian cluster

    Parameters
    ----------
    mu : 2 float array
        The mean location of the cluster.
    sigma : 2 float array
        The standard deviation of the cluster.
    N : int
        The number of localizations.

    Returns
    -------
    Nx2 float Array
        The [x1,x2] localizations .

    '''
    x1 = rnd.normal(mu[0], sigma[0], N)
    x2 = rnd.normal(mu[1], sigma[1], N)
    return np.array([x1, x2]).transpose()


def circle_2d_normal(mu, sigma, N, width):
    '''
    Generates a 2D normal circular cluster

    Parameters
    ----------
    mu : 2 float array
        The mean location of the cluster.
    sigma : 2 float array
        The standard deviation of the cluster.
    N : int
        The number of localizations
    width : float
        The width of the circle border or the spread of the Gaussian.

    Returns
    -------
    Nx2 float Array
        The [x1,x2] localizations .

    '''
    alpha = 2 * np.pi * rnd.random(N)
    rx1 = (np.abs(rnd.normal(sigma[0], width, N))) 
    rx2 = (np.abs(rnd.normal(sigma[1], width, N)))
    
    x1 = rx1 * np.sin(alpha) + mu[0]
    x2 = rx2 * np.cos(alpha) + mu[1]
    return np.array([ x1 , x2]).transpose()


def circle_2d_uniform(mu, sigma, N):
    '''
    Generates a 2D uniform circular cluster

    Parameters
    ----------
    mu : 2 float array
        The mean location of the cluster.
    sigma : 2 float array
        The standard deviation of the cluster.
    N : int
        The number of localizations

    Returns
    -------
    Nx2 float Array
        The [x1,x2] localizations .

    '''
    alpha = 2 * np.pi * rnd.random(N)
    rx1 = sigma[0] * np.sqrt(rnd.random(N)) 
    rx2 = sigma[1] * np.sqrt(rnd.random(N))
    
    x1 = rx1 * np.sin(alpha) + mu[0]
    x2 = rx2 * np.cos(alpha) + mu[1]
    return np.array([ x1 , x2]).transpose()
