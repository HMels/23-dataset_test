# generate_data.py


import numpy as np
import numpy.random as rnd

import LoadDataModules.load_data as load_data

#%% Generate Channel 
def generate_channels(path, deform, error=10, Noise=0.1, copy_channel=False, 
                           subset=1, pix_size=100):
    '''
    Parameters
    ----------
    path : str list 
        list containing the paths for 
    deform : Deform() class
        class containing the deformation parameters and functions
    error : float, optional
        Localization error. The default is 10nm.
    Noise : float, optional
        The percentage of Noise per channel. The default is 0.1.
    copy_channel : bool, optional
        Do we want to copy Channel A onto Channel B. The default is False.
        The default is True
    subset : float, optional
        The percentage of the original dataset we want to import. The default is 1
    pix_size : float, optional
        The size per pixel in nm. The default is 100.
        
    Returns
    -------
    locs_A, locs_B : Nx2 matrix float
        The actual locations of the localizations.

    '''

    if subset == 1:     # load dataset
        locs_A, locs_B = load_data.load_data_localizations(
            path, pix_size=pix_size,  alignment=(not copy_channel))
    else:               # load dataset subset
        locs_A, locs_B = load_data.load_data_subset(
            path, subset, pix_size=pix_size, alignment=(not copy_channel))
        
    if copy_channel: locs_B = locs_A.copy()
        
    # generate localization error
    locs_A = localization_error( locs_A, error )
    locs_B = localization_error( locs_B, error )
    
    img = np.empty([2,2], dtype = float)
    img[0,0] = np.min(locs_A[:,0])
    img[0,1] = np.max(locs_A[:,0])
    img[1,0] = np.min(locs_A[:,1])
    img[1,1] = np.max(locs_A[:,1])
    mid = (img[:,0] + img[:,1])/2
    
    locs_A[:,0] = locs_A[:,0] - mid[0]
    locs_B[:,0] = locs_B[:,0] - mid[0] 
    locs_A[:,1] = locs_A[:,1] - mid[1]
    locs_B[:,1] = locs_B[:,1] - mid[1]
    
    locs_B = deform.deform(locs_B)
    
    locs_A = generate_noise(locs_A, img, Noise)
    locs_B = generate_noise(locs_B, img, Noise)
    
    return locs_A, locs_B


#%% Random Generated mimic datasets
def generate_beads_mimic(deform, N=216, error=10, Noise=0.1,
                      x1_params=[-500,500], x2_params=[-300,300]):
    '''
    Generates a random channel that mimics the beads datset

    Parameters
    ----------
    deform : Deform() class
        The class containing the deformations of channel B.
    N : int, optional
        The number of localizations per channel to be generated. The default is 216.
    error : float, optional
        The error or the localizations to be generated. The default is 10nm.
    Noise : float, optional
        The percentage of uniform noise present. The default is 0.1.
    x1_params , x2_params : list, optional
        List containing the x1 and x2 system sizes. The default is [-300,300] nm.

    Returns
    -------
    locs_A, locs_B : Nx2 matrix float
        The actual locations of the localizations.

    '''
    ## channel A is uniformly random generated
    locs_A = rnd.rand(N,2)
    locs_A[:,0] = x1_params[0] + locs_A[:,0]*(x1_params[1]-x1_params[0]-50)
    locs_A[:,1] = x2_params[0] + locs_A[:,1]*(x2_params[1]-x2_params[0]-50)
    
    ## Generate channel B and give it a localization error
    locs_B = locs_A.copy()
    locs_A = localization_error( locs_A, error )
    locs_B = localization_error( locs_B, error )
    
    ## Image Parameters
    img = np.empty([2,2], dtype = float)
    img[0,0] = np.min(locs_A[:,0])
    img[0,1] = np.max(locs_A[:,0])
    img[1,0] = np.min(locs_A[:,1])
    img[1,1] = np.max(locs_A[:,1])
    mid = (img[:,0] + img[:,1])/2
    
    ## Center image
    locs_A[:,0] = locs_A[:,0] - mid[0]
    locs_B[:,0] = locs_B[:,0] - mid[0] 
    locs_A[:,1] = locs_A[:,1] - mid[1]
    locs_B[:,1] = locs_B[:,1] - mid[1]
    
    ## Generate deformation and noise
    locs_B = deform.deform(locs_B)
    locs_A = generate_noise(locs_A, img, Noise)
    locs_B = generate_noise(locs_B, img, Noise)
    
    return locs_A, locs_B


def generate_HEL1_mimic(deform, error=10, Noise=.1, Nclust=650, points_per_cluster=250, 
                        x1_params=[-255,255], x2_params=[-125,125], std_clust=7):
    '''
    Generates a mimic dataset based on the HEL1 dataset

    Parameters
    ----------
    deform : Deform() class
        The class containing the deformations of channel B.
    error : float, optional
        The error or the localizations to be generated. The default is 10nm.
    Noise : float, optional
        The percentage of uniform noise present. The default is 0.1.
    Nclust : int, optional
        The amount of clusters generated. The default is 650.
    points_per_cluster : int, optional
        The average amount of points that are generated in a cluster. The default is 250.
    x1_params , x2_params : list, optional
        List containing the x1 and x2 system sizes. The default is [-255,255] and [-125,125] nm.
    std_clust : int, optional
        The average standard deviation of a cluster. The default is 70.

    Returns
    -------
    locs_A, locs_B : Nx2 matrix float
        The actual locations of the localizations.

    '''
    
    ## Generating the locations of the Clusters
    clust_locs = rnd.rand(Nclust,2)
    clust_locs[:,0] = clust_locs[:,0]*(x1_params[1]-x1_params[0]-50)
    clust_locs[:,1] = clust_locs[:,1]*(x2_params[1]-x2_params[0]-50)
    
    
    ## Generating the Cluster Points
    locs_A = []
    i=0
    while i < Nclust:
        sigma = std_clust+30*rnd.randn(2)                           # std gets a normal random deviation
        N = int(round(points_per_cluster*(1+0.5*rnd.randn()),0))    # number of points also 
        if N>0 and sigma[0]>0 and sigma[1]>0:                       # are the points realistic
            locs_A.append(gauss_2d(clust_locs[i,:],sigma, N ))
            i+=1
            
    ## Generating more points around the clusters
    i=0
    while i < Nclust:
        sigma = 30*(std_clust+30*rnd.randn(2))
        N = int(round(points_per_cluster*(1+0.5*rnd.randn())/5,0))
        if N>0 and sigma[0]>0 and sigma[1]>0:
            locs_A.append(gauss_2d(clust_locs[i,:],sigma, N ))
            i+=1
    locs_A = np.concatenate(locs_A, axis=0)                         # add all points together
    
    ## Fit every point inside image
    locs_A[:,0] = locs_A[:,0]%(x1_params[1]-x1_params[0])
    locs_A[:,1] = locs_A[:,1]%(x2_params[1]-x2_params[0])
    
    ## Generate channel B and give it a localization error
    locs_B = locs_A.copy()
    locs_A = localization_error( locs_A, error )
    locs_B = localization_error( locs_B, error )
    
    ## Image parameters 
    img = np.empty([2,2], dtype = float)
    img[0,0] = np.min(locs_A[:,0])
    img[0,1] = np.max(locs_A[:,0])
    img[1,0] = np.min(locs_A[:,1])
    img[1,1] = np.max(locs_A[:,1])
    mid = (img[:,0] + img[:,1])/2
    
    
    ## Center image
    locs_A[:,0] = locs_A[:,0] - mid[0]
    locs_B[:,0] = locs_B[:,0] - mid[0] 
    locs_A[:,1] = locs_A[:,1] - mid[1]
    locs_B[:,1] = locs_B[:,1] - mid[1]

    ## Generate deformation and noise
    locs_B = deform.deform(locs_B)
    locs_A = generate_noise(locs_A, img, Noise)
    locs_B = generate_noise(locs_B, img, Noise)
    
    return locs_A, locs_B


#%% functions
def localization_error(locs, error = 10):
    '''
    Generates a Gaussian localization error over the localizations 

    Parameters
    ----------
    locs_: 2xN matrix float
        The actual locations of the localizations.
    error : float, optional
        The localization error in nm. The default is 10

    Returns
    -------
    locs_: Nx2 matrix float
        The actual locations of the localizations.

    '''
    if error != 0:
        N = len(locs[:,0])
        locs[:,0] += rnd.normal(0, error, N)
        locs[:,1] += rnd.normal(0, error, N)
    return locs
        

def generate_noise(locs, img, Noise):
    '''
    Parameters
    ----------
    locs_: 2xN matrix float
        The actual locations of the localizations.
    img: 2x2 array 
        containing the border values of the system
    Noise: float
        The percentage of Noice added to the system

    Returns
    -------
    locs_: Nx2 matrix float
        The actual locations of the localizations.

    '''
    if Noise != 0:
        N_Noise = int(Noise * locs.shape[0])
        
        img_size = img[:,1] - img[:,0] 
    
        Noise_loc = np.array([
            img_size[0] * ( rnd.rand( N_Noise ) -0.5) ,
            img_size[1] * ( rnd.rand( N_Noise ) -0.5)
            ])
        
        return np.append(locs, np.squeeze( Noise_loc.transpose() ), 0)
    else: 
        return locs
    

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
