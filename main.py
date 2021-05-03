# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:29:16 2021

@author: Mels

This program generates a dataset like the ones seen in super-resolution microscopy
This dataset has an induced localization error, noise, and can be manipulated with 
several deformations. Therefor, this program allows one to test several metrics for 
Channel alignment

This file consist of multiple Modules.
Main.py
|- setup_image.py               File containing the Deform class
|- generate_neighbours.py       File containing all functions for generating neighbours
|- generate_data.py             File containing everything to setup the program
|  |
|  |- load_data.py              File used to load the example dataset given
|
|- run_optimization.py          File containing the training loops
|- Minimum_Entropy.py           File containing the optimization classes 
|- output_text.py               File containing the code for the output text
|- generate_image.py            File containing the scripts to generate an image


The classes can be found in setup.py
- Cluster()

Together with the function:
- run_channel_generation()

It is optional to also run a cross-correlation program
"""

# Packages
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Modules
import setup_image
#import generate_neighbours
import generate_data
#import run_optimization
import run_optimization_stepwise as run_optimization
import output_text
import generate_image


#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)


#%% Channel Generation
## Dataset
realdata = False                                    # load real data or generate from real data
subset = 0.2                                        # percentage of original dataset
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]

## System Parameters
error = 10                                          # localization error in nm
Noise = 0.1                                         # percentage of noise

## Deformation of channel B
shift = np.array([ 14  , 17 ])                      # shift in nm
rotation = .2                                      # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
shear = np.array([0, 0])                            # shear
scaling = np.array([1, 1]) + np.array([0,0])        # scaling 
deform = setup_image.Deform(shift, rotation, shear, scaling)

## Generate Data
locs_A, locs_B = generate_data.run_channel_generation(
    path, deform, error, Noise, realdata, subset
    ) 

ch2 = tf.Variable( locs_B, dtype = tf.float32)
'''
## Generate Neighbours 
neighbour_idx = generate_neighbours.find_bright_neighbours(
    locs_A, locs_B, threshold=None, maxDistance=50)
'''

#%% Minimum Entropy
model = None
## Optimization Parameters
max_deform = 150                                    # maximum amount of deform in nm
learning_rate = 1#e-26                              # learning rate 
epochs = 2000                                        # amount of iterations of optimization

# Batches used in training 
Batch_on = False
batch_size = 4000                                   # max amount of points per batch
num_batches = np.array([3,3], dtype = int)          # amount of [x1,x2] batches

# Error Message
output_text.Info_batch( np.max([locs_A.shape[0], locs_B.shape[0]])
                       , num_batches, batch_size, Batch_on)


## Training Loop
model, loss, variables, ch2MAP = run_optimization.run_optimization(
    locs_A, locs_B,# neighbour_idx, 
    model, Batch_on,
    batch_size, num_batches, learning_rate, epochs)


#%% output
ch1, _, ch2_mapped_model = output_text.generate_output(
    locs_A, locs_B, model, variables, deform)
        

#%% generating image
## output parameters
precision = 5                                       # precision of image in nm
threshold = 100                                     # threshold for reference points

## Channel Generation
plt.close('all')
#channel1, channel2, channel2m, bounds = generate_image.generate_channel(
 #   ch1, ch2, ch2_mapped_model, precision, max_deform)
channel1, channel2, channel2m, bounds = generate_image.generate_channel(
    ch1, ch2, ch2MAP, precision, max_deform)


## Generating reference points
ref_channel1 = generate_image.reference_clust(ch1, precision * 20, 
                                              bounds, threshold)

# estimate original ch2
ch1_ref = generate_data.localization_error(locs_A, error)
channel1_ref = generate_image.generate_matrix(
    ch1_ref / precision, bounds
    )                       # generate a reference channel 

#%% Metrics

## Calculate Channel Overlap
overlap = output_text.overlap(channel1, channel2)
overlap_map = output_text.overlap(channel1, channel2m)
overlap_max = output_text.overlap(channel1, channel1_ref)

N0 = np.round(ch1.shape[0]/(1+Noise),0).astype(int)
## Calculate Average Shift
avg_shift = output_text.avg_shift(ch1[:N0,:].numpy(), ch2[:N0,:].numpy())
#avg_shift_map = output_text.avg_shift(ch1[:N0,:].numpy(), ch2_mapped_model[:N0,:].numpy())
avg_shift_map = output_text.avg_shift(ch1[:N0,:].numpy(), ch2MAP[:N0,:].numpy())
avg_shift_max = output_text.avg_shift(ch1[:N0,:].numpy(), ch1_ref[:N0,:])

print('\nThe original overlap was',overlap,'. The mapping has',overlap_map,
      '. The maximum overlap is estimated to be',overlap_max)


print('\nThe original average shift was',avg_shift,'. The mapping has',avg_shift_map,
      '. The minimum average shift is estimated to be',avg_shift_max)


#%% Output

generate_image.plot_channel(channel1, channel2, channel2m, bounds,
                            ref_channel1, precision)
print('Done')
