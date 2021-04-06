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
    ch1_data, ch2_data : the [x1, x2] locations of the localizations of 
        Channel 1 and 2.

    '''
    
    ch1 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5')
    ch2 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5')
    
    ch1_data = ch1.renderGaussianSpots(zoom = zoom)
    ch2_data = ch1.renderGaussianSpots(zoom = zoom)
    
    # Getting the data in the right format 
    # e.g. x1 being the longest row 
    ch1_data = np.transpose(ch1.pos)
    ch2_data = np.transpose(ch2.pos)
    ch1_data[[0, 1],:] = ch1_data[[1, 0],:]
    ch2_data[[0, 1],:] = ch2_data[[1, 0],:]
    
    return ch1_data, ch2_data


def load_data_localizations(zoom = 10):
    '''
    Loads the dataset localizations

    Parameters
    ----------
    zoom : int, optional
        The wanted precission of the system. The default is 10.

    Returns
    -------
    ch1_data, ch2_data : the [x1, x2] locations of the localizations of 
        Channel 1 and 2.

    '''

    ch1 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5')
    ch2 = Dataset.load('C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5')
    
    # Getting the data in the right format 
    # e.g. x1 being the longest row 
    ch1_data = np.transpose(ch1.pos)
    ch2_data = np.transpose(ch2.pos)
    ch1_data[[0, 1],:] = ch1_data[[1, 0],:]
    ch2_data[[0, 1],:] = ch2_data[[1, 0],:]
    
    return ch1_data, ch2_data

