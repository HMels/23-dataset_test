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
import run_optimization
#import MinEntropy
import MinEntropy_direct as MinEntropy
import output_fn
import generate_image

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)


#%% Channel Generation
## Dataset
realdata = True                                    # load real data or generate from real data
subset = 1                                         # percentage of original dataset
pix_size = 1
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]

## System Parameters
error = 0                                         # localization error in nm
Noise = 0.0                                         # percentage of noise

## Deformation of channel B
max_deform = 150                                    # maximum amount of deform in nm
shift = np.array([ 17  , 9 ])                      # shift in nm
rotation = .5                                       # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
shear = np.array([0.0, 0.0])                      # shear
scaling = np.array([1.0,1.0 ])                    # scaling 
deform = Deform(shift, rotation, shear, scaling)


#%% Optimization models and parameters
models = [MinEntropy.ShiftMod(), 
          MinEntropy.RotationMod(),
          MinEntropy.Poly3Mod()
          ]
optimizers = [tf.optimizers.Adagrad, 
              tf.optimizers.Adagrad, 
              tf.optimizers.Adam
              ]
learning_rates = np.array([1, 
                           1e-2,
                           1e-11
                           ])


#%% output params
plt.close('all')

hist_output = True                                  # do we want to have the histogram output
bin_width = 2                                      # Bin width in nm

plot_img = False                                     # do we want to generate a plot
precision = 5                                       # precision of image in nm
reference = False                                   # do we want to plot reference points
threshold = 100                                     # threshold for reference points


#%% Generate Data
locs_A, locs_B = generate_data.run_channel_generation(
    path, deform, error, Noise, realdata, subset, pix_size
    )

ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)


#%% Minimum Entropy
# Error Message
output_fn.Info_batch( np.max([locs_A.shape[0], locs_B.shape[0]]))


ch2_map = tf.Variable(ch2)
# training loop
mods1 = run_optimization.initiate_model(models, learning_rates, optimizers)
mods1, ch2_map = run_optimization.run_optimization(ch1, ch2_map, mods1, 30) 
print('Optimization Done!')


#%% Metrics
if hist_output:
    if realdata: N0 = ch1.shape[0]
    else: N0 = np.round(ch1.shape[0]/(1+Noise),0).astype(int)
    
    ## Calculate Average Shift errorFOV_direct
    avg1, avg2 = output_fn.errorHist_direct(ch1[:N0,:].numpy(),  ch2[:N0,:].numpy(), ch2_map[:N0,:].numpy() , bin_width)
    #avg1, avg2 = output_fn.errorHist_neighbours(ch1[:N0,:].numpy(),  ch2[:N0,:].numpy(), ch2_map[:N0,:].numpy() , bin_width)
    
    _, _ = output_fn.errorFOV_direct(ch1[:N0,:].numpy(),  ch2[:N0,:].numpy(), ch2_map[:N0,:].numpy())
    #_, _ = output_fn.errorFOV_neighbours(ch1[:N0,:].numpy(),  ch2[:N0,:].numpy(), ch2_map[:N0,:].numpy())
    
    print('\nI: The original average distance was', avg1,'. The mapping has', avg2)


#%% generating image
if plot_img:
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


