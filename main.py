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
|- generate_data.py             File containing everything to setup the program
|  |
|  |- load_data.py              File used to load the example dataset given
|
|- pre_alignment.py             File containing MinEntr functions for pre-aligning
|- run_optimization.py          File containing the training loops
|- Minimum_Entropy.py           File containing the optimization classes 
   |- generate_neighbours.py    File containing all functions for generating neighbours
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

# Classes
from setup_image import Deform

# Modules
import generate_data
import pre_alignment
import run_optimization
import Minimum_Entropy
import output_text
import generate_image

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)


#%% Channel Generation
## Dataset
realdata = True                                    # load real data or generate from real data
subset = 1                                        # percentage of original dataset
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]

## System Parameters
error = 0                                         # localization error in nm
Noise = 0.0                                         # percentage of noise

## Deformation of channel B
max_deform = 150                                    # maximum amount of deform in nm
shift = np.array([ 17  , 19 ])                      # shift in nm
rotation = .2                                       # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
shear = np.array([0.0, 0.0])                      # shear
scaling = np.array([1.0,1.0 ])                    # scaling 
deform = Deform(shift, rotation, shear, scaling)


#%% Optimization models and parameters
models = [Minimum_Entropy.ShiftMod('shift'), 
          Minimum_Entropy.RotationMod('rotation'),
          Minimum_Entropy.Poly3Mod('polynomial')
          ]
optimizers = [tf.optimizers.Adagrad, 
              tf.optimizers.Adagrad, 
              tf.optimizers.Adam
              ]
learning_rates = np.array([1, 
                           1e-2,
                           1e-18
                           ])

# Batches used in training 
Batch_on = False
batch_size = 4000                                   # max amount of points per batch
num_batches = np.array([3,3], dtype = int)          # amount of [x1,x2] batches

plot = False                                        # do we want to generate a plot


#%% Generate Data
locs_A, locs_B = generate_data.run_channel_generation(
    path, deform, error, Noise, realdata, subset
    )

ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)


#%% Minimum Entropy
# Error Message
output_text.Info_batch( np.max([locs_A.shape[0], locs_B.shape[0]])
                       , num_batches, batch_size, Batch_on)

# pre-aligning the data with MinEntropy
ch2_map , mods0 = pre_alignment.align(ch1, ch2, mods=None, maxDistance=150)
#ch2_map = tf.Variable(ch2)

# training loop
mods1 = Minimum_Entropy.initiate_model(models, learning_rates, optimizers)
mods1, ch2_map = run_optimization.run_optimization(ch1, ch2_map, mods1, 50) 


mods2 = Minimum_Entropy.initiate_model(models, learning_rates, optimizers)
mods2, ch2_map = run_optimization.run_optimization(ch1, ch2_map, mods2, 50) 
'''
mods2 = Minimum_Entropy.initiate_model(Minimum_Entropy.Poly3Mod('polynomial'), 
                                      1e-17, tf.optimizers.Adam)
mods2, ch2_map = run_optimization.run_optimization(ch1, ch2_map, mods2, 50) 
'''
print('Optimization Done!')


#%% Metrics
plt.close('all')
if realdata: N0 = ch1.shape[0]
else: N0 = np.round(ch1.shape[0]/(1+Noise),0).astype(int)

## Calculate Average Shift
avg1, avg2 = output_text.precision_distr(ch1[:N0,:].numpy(),  ch2[:N0,:].numpy(), 
                                         ch2_map[:N0,:].numpy() , bin_width = 20)

print('\nI: The original average distance was', avg1,'. The mapping has', avg2)


#%% generating image
if plot:
    precision = 500                                       # precision of image in nm
    reference = False                                   # do we want to plot reference points
    threshold = 100                                     # threshold for reference points

    ## Channel Generation
    channel1, channel2, channel2m, bounds = generate_image.generate_channel(
        ch1, ch2, ch2_map, precision, max_deform)
    
    
    ## Generating reference points
    ref_channel1 = generate_image.reference_clust(ch1, precision * 20, 
                                                  bounds, threshold, reference)


    # estimate original ch2
    ch1_ref = generate_data.localization_error(locs_A, error)
    channel1_ref = generate_image.generate_matrix(
        ch1_ref / precision, bounds
        )
    
    generate_image.plot_channel(channel1, channel2, channel2m, bounds,
                            ref_channel1, precision, reference)

print('Done')


