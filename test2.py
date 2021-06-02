# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:21:42 2021

@author: Mels
"""
import tensorflow as tf
import numpy as np 

from setup_image import Deform
import generate_neighbours
import generate_data
#%% Channel Generation
## Dataset
realdata = True                                    # load real data or generate from real data
direct = True                                       # True if data is coupled
subset = 1                                         # percentage of original dataset
pix_size = 1
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]

## System Parameters
error = 0.1                                         # localization error in nm
Noise = 0.0                                         # percentage of noise

## Deformation of channel B
max_deform = 150                                    # maximum amount of deform in nm
shift = np.array([ 13  , 9 ])                      # shift in nm
rotation = .05                                      # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
shear = np.array([0.003, 0.002])                      # shear
scaling = np.array([1.0004,1.0003 ])                    # scaling 
deform = Deform(shift, rotation, shear, scaling)

#%% Generate Data
locs_A, locs_B = generate_data.run_channel_generation(
    path, deform, error, Noise, realdata, subset, pix_size
    )
ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)

#%%
gridsize=10
ch2_input = tf.Variable(ch2/gridsize, trainable=False)
        
CP_idx = tf.cast(tf.stack(
    [( ch2_input[:,0]-tf.reduce_min(tf.floor(ch2_input[:,0]))+1)//1 , 
     ( ch2_input[:,1]-tf.reduce_min(tf.floor(ch2_input[:,1]))+1)//1 ], 
    axis=1), dtype=tf.int32)

neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
ch2_input.numpy(), ch2_input.numpy(), maxDistance=10/gridsize, threshold=None)
nn1 = ch1[:,None]
#nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
nn2 = ch2[:,None]
#nn2 = tf.Variable( neighbours_B, dtype = tf.float32)

#%%        
x1_grid = tf.range(tf.reduce_min(tf.floor(ch2_input[:,0])) -1,
                   tf.reduce_max(tf.floor(ch2_input[:,0])) +3, 1)
x2_grid =  tf.range(tf.reduce_min(tf.floor(ch2_input[:,1]))-1,
                    tf.reduce_max(tf.floor(ch2_input[:,1])) +3, 1)
CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
CP_idx_nn = tf.cast(tf.stack(
    [( nn2[:,:,0]-tf.reduce_min(tf.floor(nn2[:,:,0]))+1)//1 , 
     ( nn2[:,:,1]-tf.reduce_min(tf.floor(nn2[:,:,1]))+1)//1 ], 
    axis=2), dtype=tf.int32)  

q00 = tf.gather_nd(CP_locs, CP_idx_nn+[-1,-1])  # q_k