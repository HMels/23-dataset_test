# setup.py
'''
File that contains everything to setup the program

The functions are:
- run_channel_generation_distribution()
    which generates the localization dataset according to a certain distributions
    and clusters


The classes are:
- Cluster()
    The class containing the information of a certain cluster
- Image()
    The class containing the information of the whole system 
'''

import numpy as np
import generate_dataset
import dataset_manipulation
import load_data

#%% functions
def run_channel_generation_distribution(cluster, img_param, angle = 0, shift = np.array([0,0]),
                           error = 0.1):
    '''
    Generates a Channel dataset via a certain distribution
    
    Parameters
    ----------
    cluster : Cluster() array
        Class containing the data of the clusters needed to be generated.
    img_param : Image()
        Class containing the data of the image.
    angle : float, optional
        Angle of rotation between channel A and B. The default is 0.
    shift : float, optional
        Shift between channel A and B. The default is np.array([0,0]).
    error : float, optional
        Localization error in pixels. The default is 0.1.

    Returns
    -------
    channel_A, channel_B : NxN matrix int
        Matrix containing the image of channel A or B.
    localizations_A, localizations_B : 2xN matrix float
        The actual locations of the localizations.

    '''
    # Generate the true localizations from the cluster distribution
    localizations_A = generate_dataset.generate_localizations( cluster, img_param )
    localizations_B = localizations_A.copy()
    
    # Generate localization error 
    localizations_A = generate_dataset.localization_error( localizations_A, img_param, error )
    localizations_B = generate_dataset.localization_error( localizations_B, img_param, error )
    
    # Induce shift and rotation in Channel B
    if angle:
        localizations_B = dataset_manipulation.rotation( localizations_B, angle, img_param )
    if shift.all():
        localizations_B = dataset_manipulation.shift( localizations_B, shift )

    # Generate Noise
    localizations_A = generate_dataset.generate_noise(img_param, localizations_A)
    localizations_B = generate_dataset.generate_noise(img_param, localizations_B)
    
    # Generate the Channels in matrix
    channel_A = generate_dataset.generate_channel(img_param, localizations_A)
    channel_B = generate_dataset.generate_channel(img_param, localizations_B)
    
    return channel_A, channel_B, localizations_A, localizations_B


def run_channel_generation_realdata(img_param, angle = 0, shift = np.array([0,0]),
                           error = 0.1):
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

    Returns
    -------
    channel_A, channel_B : NxN matrix int
        Matrix containing the image of channel A or B.
    localizations_A, localizations_B : 2xN matrix float
        The actual locations of the localizations.

    '''
    
    # Generate the true localizations from the cluster distribution
    localizations_A, _ = load_data.load_data_localizations(img_param.zoom)
    localizations_B = localizations_A.copy()
    
    # Thinning out the dataset
    N = len(localizations_A[0,:])
    index = np.random.choice(N, int(0.1*N), replace=False)  
    localizations_A = localizations_A[:, index]
    localizations_B = localizations_B[:, index]
    
    # calculating new boundaries
    max_x1 = np.max([ localizations_A[0,:].max(), localizations_B[0,:].max() ])
    max_x2 = np.max([ localizations_A[1,:].max(), localizations_B[1,:].max() ])
    min_x1 = np.max([ localizations_A[0,:].min(), localizations_B[0,:].min() ])
    min_x2 = np.max([ localizations_A[1,:].min(), localizations_B[1,:].min() ])
    img_param.img_size_x1 = int( (max_x1 - min_x1) / img_param.pix_size_zoom() )
    img_param.img_size_x2 = int( (max_x2 - min_x2) / img_param.pix_size_zoom() )

    # Generate localization error 
    localizations_A = generate_dataset.localization_error( localizations_A, img_param, error )
    localizations_B = generate_dataset.localization_error( localizations_B, img_param, error )
    
    # Induce shift and rotation in Channel B
    if angle:
        localizations_B = dataset_manipulation.rotation( localizations_B, angle, img_param )
    if shift.all():
        localizations_B = dataset_manipulation.shift( localizations_B, shift )

    # Generate Noise
    localizations_A = generate_dataset.generate_noise(img_param, localizations_A)
    localizations_B = generate_dataset.generate_noise(img_param, localizations_B)
    
    # Generate the Channels in matrix
    channel_A = generate_dataset.generate_channel(img_param, localizations_A)
    channel_B = generate_dataset.generate_channel(img_param, localizations_B)
    
    return channel_A, channel_B, localizations_A, localizations_B, img_param

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
    

class Image():
    '''
    The class that defines the Image and it's parameters 
    contains the objects
    zoom(): the level of precission in [nm]
    pix_size(): size of a pixel in [nm]
    pix_size_zoom(): pix_size in units of [zoom]
    img_size(): an array containing size of the image in [pix]
    img_size_zoom(): img_size in units of [zoom]
    Noise(): the number of false localizations due to noise 
    '''
    def __init__(self, zoom, pix_size, img_size_x1,
                 img_size_x2, Noise ):
        self.zoom = zoom                        # nm
        self.pix_size = pix_size                # nm
        self.img_size_x1 = int(img_size_x1)     # pix
        self.img_size_x2 = int(img_size_x2)     # pix
        self.Noise = Noise
        
    def zoom(self): return self.zoom
    def pix_size_nm(self): return self.pix_size 
    def pix_size_zoom(self): return self.pix_size / self.zoom
    
    def img_size_pix(self): return np.array([self.img_size_x1, self.img_size_x2])
    def img_size_zoom(self): return self.img_size_pix() * self.pix_size_zoom()
    def img_size_nm(self): return self.img_size_pix() * self.pix_size
    
    def Noise(self): return self.Noise
    