# load_data
from photonpy import Dataset
import numpy as np


def load_data_GaussianSpots(path, alignment = True, zoom = 10, pix_size = 100):
    '''
    Loads the dataset before localization (zo localizations have gaussian
                                           spots around them)

    Parameters
    ----------
    path : list
        list containing the paths of ch1 and ch2
    alignment : bool, optional
        True if we want to do a FFT cross-correlation alignment beforehand
    zoom : int, optional
        The wanted precission of the system. The default is 10.
    pix_size : float, optional
        size of the pixels in nm

    Returns
    -------
    ch1_data, ch2_data : Nx2 float array
        The [x1, x2] locations of the localizations of Channel 1 and 2.

    '''
    if len(path)==1:
        print('Loading dataset... \n Grouping...')
        ds = Dataset.load(path[0],saveGroups=True)
        ch1 = ds[ds.group==0]
        ch2 = ds[ds.group==1]
    elif len(path)==2:
        print('Loading dataset...')
        ch1 = Dataset.load(path[0])
        ch2 = Dataset.load(path[1])
    else:
        print('Error: Path invalid for current coupled setting!')
    
    if alignment:
        print('Alignning both datasets')
        Dataset.align(ch1, ch2)
    

    ch1_data = ch1.renderGaussianSpots(zoom = zoom)
    ch2_data = ch2.renderGaussianSpots(zoom = zoom)
    
    # Getting the data in the right format 
    ch1_data[:,[0, 1]] = ch1_data[:,[1, 0]]
    ch2_data[:,[0, 1]] = ch2_data[:,[1, 0]]
    
    return ch1_data*pix_size, ch2_data*pix_size


def load_data_localizations(path, alignment = True, pix_size = 100):
    '''
    Loads the dataset localizations
    
    Parameters
    ----------
    path : list
        list containing the paths of ch1 and ch2
    alignment : bool, optional
        True if we want to do a FFT cross-correlation alignment beforehand
    pix_size : float, optional
        size of the pixels in nm. The default is 100.

    Returns
    -------
    ch1_data, ch2_data : Nx2 float array
        The [x1, x2] locations of the localizations of Channel 1 and 2.

    '''
    if len(path)==1:
        print('Loading dataset... \n Grouping...')
        ds = Dataset.load(path[0],saveGroups=True)
        ch1 = ds[ds.group==0]
        ch2 = ds[ds.group==1]
    elif len(path)==2:
        print('Loading dataset...')
        ch1 = Dataset.load(path[0])
        ch2 = Dataset.load(path[1])
    else:
        print('Error: Path invalid for current group setting!')
    
    if alignment:
        print('Alignning both datasets')
        Dataset.align(ch1, ch2)
    
    
    # Getting the data in the right format 
    # e.g. x1 being the longest row 
    ch1_data = ch1.pos
    ch2_data = ch2.pos
    ch1_data[:,[0, 1]] = ch1_data[:,[1, 0]]
    ch2_data[:,[0, 1]] = ch2_data[:,[1, 0]]
    
    return ch1_data*pix_size, ch2_data*pix_size


def load_data_subset(path, subset = 0.2, alignment = True, pix_size = 100):
    '''
    Loads a subset of the dataset localizations
    
    Parameters
    ----------
    path : list
        list containing the paths of ch1 and ch2
    subset : float, optional
        The size of the subset relative to the total set. The default is 0.2.
    alignment : bool, optional
        True if we want to do a FFT cross-correlation alignment beforehand
    pix_size : float, optional
        size of the pixels in nm. The default is 100

    Returns
    -------
    ch1_data, ch2_data : Nx2 float array
        The [x1, x2] locations of the localizations of Channel 1 and 2.

    '''
    if len(path)==1:
        print('Loading dataset... \n Grouping...')
        ds = Dataset.load(path[0],saveGroups=True)
        ch1 = ds[ds.group==0]
        ch2 = ds[ds.group==1]
    elif len(path)==2:
        print('Loading dataset...')
        ch1 = Dataset.load(path[0])
        ch2 = Dataset.load(path[1])
    else:
        print('Error: Path invalid for current group setting!')
    
    if alignment:
        print('Alignning both datasets')
        Dataset.align(ch1, ch2)
    
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
        
    return ch1_data*pix_size, ch2_data*pix_size
