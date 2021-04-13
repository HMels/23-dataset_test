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
def run_channel_generation_distribution(cluster, angle = 0, shift = np.array([0,0]),
                           error = 0.1, Noise = 0.1):
    '''
    Generates a Channel dataset via a certain distribution
    
    Parameters
    ----------
    cluster : Cluster() array
        Class containing the data of the clusters needed to be generated.
    angle : float, optional
        Angle of rotation between channel A and B. The default is 0.
    shift : float, optional
        Shift between channel A and B. The default is np.array([0,0]).
    error : float, optional
        Localization error in pixels. The default is 0.1.
    Noise : float, optional
        The percentage of Noise per channel. The default is 0.1.

    Returns
    -------
    localizations_A, localizations_B : Nx2 matrix float
        The actual locations of the localizations.

    '''
    # Generate the true localizations from the cluster distribution
    localizations_A = generate_dataset.generate_localizations( cluster )
    localizations_B = localizations_A.copy()
    
    # Generate localization error 
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
    
    # Induce shift and rotation in Channel B
    if angle:
        localizations_B = dataset_manipulation.rotation( localizations_B, angle )
    if shift.all():
        localizations_B = dataset_manipulation.shift( localizations_B, shift )

    # Generate Noise
    localizations_A = generate_dataset.generate_noise(localizations_A, img, Noise)
    localizations_B = generate_dataset.generate_noise(localizations_B, img, Noise)
    
    return localizations_A, localizations_B


def run_channel_generation_realdata(img_param, angle = 0, shift = np.array([0,0]),
                           error = 0.1, batch_size = 0.1, Noise = 0.1):
    '''
    Generates a Channel dataset via real data

    Parameters
    ----------
    img_param : Image()
        Class containing the data of the image.
    angle : float, optional
        Angle of rotation between channel A and B. The default is 0.
    shift : float, optional
        Shift between channel A and B. The default is np.array([0,0]).
    error : float, optional
        Localization error in pixels. The default is 0.1.
    batch_size : float, optional
        The size of the subset for which the mapping will be calculated. The default is 0.1
    Noise : float, optional
        The percentage of Noise per channel. The default is 0.1.
        
    Returns
    -------
    localizations_A, localizations_B : 2xN matrix float
        The actual locations of the localizations.

    '''
    
    # Generate the true localizations from the cluster distribution
    localizations_A, _ = load_data.load_data_localizations()
    localizations_B = localizations_A.copy()
    
    # Thinning out the dataset
    N = len(localizations_A[:,0])
    index = np.random.choice(N, int(0.01*N), replace=False)  
    localizations_A = localizations_A[index, :]
    localizations_B = localizations_B[index, :]
    
    # Generate localization error 
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
    
    # Induce shift and rotation in Channel B
    if angle:
        localizations_B = dataset_manipulation.rotation( localizations_B, angle )
    if shift.all():
        localizations_B = dataset_manipulation.shift( localizations_B, shift )

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
    
