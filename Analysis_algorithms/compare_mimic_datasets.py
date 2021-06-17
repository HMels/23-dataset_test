# compare_mimic_datasets.py
"""
This program is mainly build to test the HEL1 and Beads Mimic dataset generators 
on and compare them to eachother
"""

import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt

import OutputModules.generate_image as generate_image
import LoadDataModules.generate_data as generate_data
import LoadDataModules.load_data as load_data
from LoadDataModules.Deform import Deform


#%% plot function
def plot_channel(channel1, channel2, channel3, precision, bounds):
    axis = np.array([ bounds[1,:], bounds[0,:]]) * precision
    axis = np.reshape(axis, [1,4])[0]

    # plotting all channels
    plt.figure()
    plt.subplot(131)
    plt.imshow(channel1, extent = axis)
    plt.title('Randomly generated Dataset (Mimic)')
    
    plt.subplot(132)
    plt.imshow(channel2, extent = axis)
    plt.title('Original Channel A')
    
    plt.subplot(133)
    plt.imshow(channel3, extent = axis)
    plt.title('Original Channel B')


#%% Beads 
shift = np.array([ 20  , 20 ]) + 10*rnd.randn(2)                     # shift in nm
rotation = 0.2*rnd.randn(1)                 # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
deform = Deform(shift, rotation)

locs_A, locs_B = generate_data.generate_beads_mimic(216, deform, error=0, Noise=0)

path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]
locs_A_original, locs_B_original = load_data.load_data_localizations(
            path, pix_size=1,  alignment=True)

locs_A_original[:,0]-=np.max(locs_A_original[:,0])/2
locs_A_original[:,1]-=np.max(locs_A_original[:,1])/2
locs_B_original[:,0]-=np.max(locs_B_original[:,0])/2
locs_B_original[:,1]-=np.max(locs_B_original[:,1])/2


precision=5 
## Channel Generation
channel1, channel2, channel3, bounds = generate_image.generate_channel(
    locs_A, locs_A_original, locs_B_original, precision)
plot_channel(channel1, channel2, channel3, precision, bounds)


#%% HEL1 Mimic
shift = np.array([ 20  , 20 ]) + 10*rnd.randn(2)                     # shift in nm
rotation = 0.2*rnd.randn(1)                 # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
deform = Deform(shift, rotation)
        
locs_A, locs_B = generate_data.generate_HEL1_mimic(deform=deform, error=0, Noise=0)


path=[ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
locs_A_original, locs_B_original = load_data.load_data_localizations(
            path, pix_size=100,  alignment=True)

locs_A_original[:,0]-=np.max(locs_A_original[:,0])/2
locs_A_original[:,1]-=np.max(locs_A_original[:,1])/2
locs_B_original[:,0]-=np.max(locs_B_original[:,0])/2
locs_B_original[:,1]-=np.max(locs_B_original[:,1])/2


precision=100 
## Channel Generation
channel1, channel2, channel3, bounds = generate_image.generate_channel(
    locs_A, locs_A_original, locs_B_original, precision)
plot_channel(channel1, channel2, channel3, precision, bounds)