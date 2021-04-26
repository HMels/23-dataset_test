# generate_data.py


import numpy as np
import numpy.random as rnd

import dataset_manipulation
import load_data

#%% functions
def run_channel_generation(path, shift = np.array([0,0]), angle = 0, 
                                        shear = np.array([0,0]), scaling = np.array([1,1]), 
                                        error = 10, Noise = 0.1, realdata = True, 
                                        subset = 1):
    '''
    Parameters
    ----------
    path : str list 
        list containing the paths for 
    angle : float, optional
        Angle of rotation between channel A and B in radians. The default is 0.
    shift : float, optional
        Shift between channel A and B in pix. The default is np.array([0,0]).
    shear : float, optional
        Shear between channel A and B. The default is np.array([0,0])
    scaling : float, optional
        Scaling between channel A and B. The default is np.array([1,1])
    error : float, optional
        Localization error in nm. The default is 10.
    Noise : float, optional
        The percentage of Noise per channel. The default is 0.1.
    realdata : bool, optional
        True if we want to use real data, False if we base our dataset on real data
        The default is True
    subset : float, optional
        The percentage of the original dataset we want to import. The default is 1
        
    Returns
    -------
    locs_A, locs_B : Nx2 matrix float
        The actual locations of the localizations.

    '''

    if subset == 1:     # load dataset
        locs_A, locs_B = load_data.load_data_localizations(
            path, alignment = realdata)
    else:               # load dataset subset
        locs_A, locs_B = load_data.load_data_subset(
            path, subset, alignment = realdata)
        
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
    
    if not realdata:      # Induce deformation in Channel B
        locs_B = dataset_manipulation.complex_translation(locs_B, 
                                                                   shift, angle, 
                                                                   shear, scaling)
        # Generate Noise
        locs_A = generate_noise(locs_A, img, Noise)
        locs_B = generate_noise(locs_B, img, Noise)
    
    return locs_A, locs_B


def localization_error(locs_, error = 10):
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
    N = len(locs_[:,0])
    locs_[:,0] += rnd.normal(0, error, N)
    locs_[:,1] += rnd.normal(0, error, N)
    return locs_
        

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
    
    img_size = img[1,:] - img[0,:] 

    Noise_loc = np.array([
        img_size[0] * ( rnd.rand( N_Noise ) -0.5) ,
        img_size[1] * ( rnd.rand( N_Noise ) -0.5)
        ])
    
    return np.append(locs_, np.squeeze( Noise_loc.transpose() ), 0)
    
