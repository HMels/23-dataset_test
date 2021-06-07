# generate_data.py


import numpy as np
import numpy.random as rnd

import load_data

#%% Generate Channel fuctions
def generate_channels(path, deform, error=10, Noise=0.1, realdata=True, 
                           subset=1, pix_size=100):
    '''
    Parameters
    ----------
    path : str list 
        list containing the paths for 
    deform : Deform() class
        class containing the deformation parameters and functions
    error : float, optional
        Localization error in nm. The default is 10.
    Noise : float, optional
        The percentage of Noise per channel. The default is 0.1.
    realdata : bool, optional
        True if we want to use real data, False if we base our dataset on real data
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
            path, pix_size=pix_size,  alignment=realdata)
    else:               # load dataset subset
        locs_A, locs_B = load_data.load_data_subset(
            path, subset, pix_size=pix_size, alignment=realdata)
        
    if not realdata:    # generate channel based on dataset with induced error 
        locs_B = locs_A.copy()
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
    
    if not realdata:      # Induce deformation and noise in Channel B
        locs_B = deform.deform(locs_B)
        locs_A = generate_noise(locs_A, img, Noise)
        locs_B = generate_noise(locs_B, img, Noise)
    
    return locs_A, locs_B


def generate_channels_random(N, deform, error=10, Noise=0.1,
                      x1_params=[-500,500], x2_params=[-300,300]):
    '''
    Generates a random channel

    Parameters
    ----------
    N : int
        The number of localizations per channel to be generated.
    deform : Deform() class
        The class containing the deformations of channel B.
    error : float, optional
        The error or the localizations to be generated. The default is 10.
    Noise : float, optional
        The percentage of uniform noise present. The default is 0.1.
    x1_params , x2_params : list, optional
        List containing the x1 and x2 system sizes. The default is [-300,300].

    Returns
    -------
    locs_A, locs_B : Nx2 matrix float
        The actual locations of the localizations.

    '''
    
    locs_A = rnd.rand(N,2)
    locs_A[:,0] = x1_params[0] + locs_A[:,0]*(x1_params[1]-x1_params[0]-50)
    locs_A[:,1] = x2_params[0] + locs_A[:,1]*(x2_params[1]-x2_params[0]-50)
    
    locs_B = locs_A.copy()
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
    N = len(locs[:,0])
    locs[:,0] += rnd.normal(0, error, N)
    locs[:,1] += rnd.normal(0, error, N)
    return locs
        

def generate_noise(locs_, img, Noise):
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
    N_Noise = int(Noise * locs_.shape[0])
    
    img_size = img[:,1] - img[:,0] 

    Noise_loc = np.array([
        img_size[0] * ( rnd.rand( N_Noise ) -0.5) ,
        img_size[1] * ( rnd.rand( N_Noise ) -0.5)
        ])
    
    return np.append(locs_, np.squeeze( Noise_loc.transpose() ), 0)
    
