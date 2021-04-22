# setup.py
'''
File that contains everything to setup the program

The functions are:
- run_channel_generation_distribution()
    which generates the localization dataset according to a certain distributions
    and clusters
- run_channel_generation_realdata()
    which generates the localization dataset according to an existing dataset


The classes are:
- Cluster()
    The class containing the information of a certain cluster
'''

import numpy as np
import generate_dataset
import dataset_manipulation
import load_data

#%% functions
def run_channel_generation(cluster, shift = np.array([0,0]), angle = 0, 
                                        shear = np.array([0,0]), scaling = np.array([1,1]), 
                                        error = .1, Noise = 0.1, realdata = True, autodeform = True):
    '''
    generate Channel
    
    Parameters
    ----------
    cluster : Cluster() array
        Class containing the data of the clusters needed to be generated.
    angle : float, optional
        Angle of rotation between channel A and B in radians. The default is 0.
    shift : float, optional
        Shift between channel A and B in pix. The default is np.array([0,0]).
    shear : float, optional
        Shear between channel A and B. The default is np.array([0,0])
    scaling : float, optional
        Scaling between channel A and B. The default is np.array([1,1])
    error : float, optional
        Localization error in pix. The default is .1.
    Noise : float, optional
        The percentage of Noise per channel. The default is 0.1.
    realdata : bool, optional
        True if we want to use real data, False if based on cluster distribution
    autodeform : bool, optional
        True if we want to use a known deformation, False if we want to use the real dataset

    Returns
    -------
    localizations_A, localizations_B : Nx2 matrix float
        The actual locations of the localizations.

    '''
    if realdata and autodeform: # generate channel based on dataset with induced error 
        #localizations_A, _ = load_data.load_data_localizations()
        localizations_A, _ = load_data.load_data_subset(subset = 0.2)
        localizations_B = localizations_A.copy()
        
        localizations_A = generate_dataset.localization_error( localizations_A, error )
        localizations_B = generate_dataset.localization_error( localizations_B, error )
        
    elif realdata and not autodeform: # generate channel of real dataset
        #localizations_A, localizations_B = load_data.load_data_localizations()
        localizations_A, localizations_B = load_data.load_data_subset(subset = 0.5)
        
    else: # generate channel with induced error based on clusters
        localizations_A = generate_dataset.generate_localizations( cluster ) # locs in nm
        localizations_B = localizations_A.copy()
        
        localizations_A = generate_dataset.localization_error( localizations_A, error )
        localizations_B = generate_dataset.localization_error( localizations_B, error )

    img = np.empty([2,2], dtype = float)
    img[0,0] = np.min(localizations_A[:,0])
    img[1,0] = np.max(localizations_A[:,0])
    img[0,1] = np.min(localizations_A[:,1])
    img[1,1] = np.max(localizations_A[:,1])
    mid = (img[0,:] + img[1,:])/2
    
    localizations_A[:,0] = localizations_A[:,0] - mid[0]
    localizations_B[:,0] = localizations_B[:,0] - mid[0] 
    localizations_A[:,1] = localizations_A[:,1] - mid[1]
    localizations_B[:,1] = localizations_B[:,1] - mid[1]
    
    if autodeform:
        # Induce deformation in Channel B
        localizations_B = dataset_manipulation.complex_translation(localizations_B, 
                                                                   shift, angle, 
                                                                   shear, scaling)
        # Generate Noise
        localizations_A = generate_dataset.generate_noise(localizations_A, img, Noise)
        localizations_B = generate_dataset.generate_noise(localizations_B, img, Noise)
    
    return localizations_A, localizations_B


#%% classes
class Cluster():
    '''
    The class that defines clusters. 
    Characteristic are the [x1, x2] location, [x1, x2] standard deviation
    and N points    
    '''
    def __init__(self, loc_x1, loc_x2, std_x1, std_x2, N):
        self.loc_x1 = loc_x1        # pix
        self.loc_x2 = loc_x2        # pix
        self.std_x1 = std_x1        # pix
        self.std_x2 = std_x2        # pix
        self.N = N

    def loc(self): return np.array([self.loc_x1, self.loc_x2])
    def std(self): return np.array([self.std_x1, self.std_x2])
    def num(self): return self.N
    
