# load_data
from photonpy import Dataset
import numpy as np

def load_data_GaussianSpots(zoom = 10):
    '''
    Loads the dataset before localization (zo localizations have gaussian
                                           spots around them)

    Parameters
    ----------
    zoom : int, optional
        The wanted precission of the system. The default is 10.

    Returns
    -------
    ch1_data, ch2_data : Nx2 float array
        The [x1, x2] locations of the localizations of Channel 1 and 2.

    '''
    
    ch1 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5')
    ch2 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5')
    
    ch1_data = ch1.renderGaussianSpots(zoom = zoom)
    ch2_data = ch2.renderGaussianSpots(zoom = zoom)
    
    # Getting the data in the right format 
    ch1_data[:,[0, 1]] = ch1_data[:,[1, 0]]
    ch2_data[:,[0, 1]] = ch2_data[:,[1, 0]]
    
    return ch1_data, ch2_data


def load_data_localizations():
    '''
    Loads the dataset localizations

    Returns
    -------
    ch1_data, ch2_data : Nx2 float array
        The [x1, x2] locations of the localizations of Channel 1 and 2.

    '''

    ch1 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5')
    ch2 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5')
    
    # Getting the data in the right format 
    # e.g. x1 being the longest row 
    ch1_data = ch1.pos
    ch2_data = ch2.pos
    ch1_data[:,[0, 1]] = ch1_data[:,[1, 0]]
    ch2_data[:,[0, 1]] = ch2_data[:,[1, 0]]
    
    return ch1_data, ch2_data


def load_data_subset(subset = 0.1):
    '''
    Loads a subset of the dataset localizations
    
    Parameters
    ----------
    subset : float, optional
        The size of the subset relative to the total set. The default is 0.1.

    Returns
    -------
    ch1_data, ch2_data : Nx2 float array
        The [x1, x2] locations of the localizations of Channel 1 and 2.

    '''

    ch1 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5')
    ch2 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5')
    
    # Getting the data in the right format 
    # e.g. x1 being the longest row 
    ch1_data = ch1.pos
    ch2_data = ch2.pos
    ch1_data[:,[0, 1]] = ch1_data[:,[1, 0]]
    ch2_data[:,[0, 1]] = ch2_data[:,[1, 0]]
    
    img = np.empty([2,2], dtype = float)        # calculate borders of system
    img[0,0] = np.min(ch1_data[:,0])
    img[1,0] = np.max(ch1_data[:,0])
    img[0,1] = np.min(ch1_data[:,1])
    img[1,1] = np.max(ch1_data[:,1])
    size_img = img[1,:] - img[0,:]
    mid = (img[1,:] + img[0,:])/2
    
    l_grid = mid - np.array([ subset*size_img[0], subset*size_img[1] ])/2
    r_grid = mid + np.array([ subset*size_img[0], subset*size_img[1] ])/2
    
    indx1 = np.argwhere( (ch1_data[:,0] >= l_grid[0]) * (ch1_data[:,1] >= l_grid[1])
                        * (ch1_data[:,0] <= r_grid[0]) * (ch1_data[:,1] <= r_grid[1]) )[:,0]
    
    indx2 = np.argwhere( (ch2_data[:,0] >= l_grid[0]) * (ch2_data[:,1] >= l_grid[1])
                        * (ch2_data[:,0] <= r_grid[0]) * (ch2_data[:,1] <= r_grid[1]) )[:,0]

    ch1_data = ch1_data[ indx1, : ]
    ch2_data = ch2_data[ indx2, : ]
        
    return ch1_data, ch2_data