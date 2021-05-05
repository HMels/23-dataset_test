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
from run_optimization import Models
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
realdata = False                                    # load real data or generate from real data
subset = .2                                        # percentage of original dataset
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]

## System Parameters
error = 0                                          # localization error in nm
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
          Minimum_Entropy.RotationMod('rotation')#,
          #Minimum_Entropy.Poly3Mod('polynomial')
          ]
optimizers = [tf.optimizers.Adagrad, 
              tf.optimizers.Adagrad#, 
              #tf.optimizers.Adam
              ]
learning_rates = [1, 1e-2]#, 1e-14]

# Batches used in training 
Batch_on = False
batch_size = 4000                                   # max amount of points per batch
num_batches = np.array([3,3], dtype = int)          # amount of [x1,x2] batches

search_area = [35]


#%% output parameters
precision = 5                                       # precision of image in nm
reference = False                                   # do we want to plot reference points
threshold = 100                                     # threshold for reference points


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

mods = []
for i in range(len(models)):
    mods.append( Models(model=models[i], learning_rate = learning_rates[i], 
                        opt=optimizers[i] ))
    mods[i].var = mods[i].model.trainable_variables


# pre-aligning the data with MinEntropy
ch2_map , mods1 = pre_alignment.align(ch1, ch2, mods=None, maxDistance=50)

## training for decreasing search areas
for i in range(len(search_area)):
    # training loop
    mods, ch2_map = run_optimization.run_optimization(ch1, ch2_map, mods, search_area[i]) 
    
    #reset the training loop
    if i != len(search_area):
        print('Optimization ',i+1,'done!\n')
        for mod in mods:
            mod.endloop = False
            mod.reset_learning_rate(mod.learning_rate/10)
print('Optimization Done!')


#%% generating image
## Channel Generation
plt.close('all')
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


#%% Metrics
## Calculate Channel Overlap
overlap = output_text.overlap(channel1, channel2)
overlap_map = output_text.overlap(channel1, channel2m)
overlap_max = output_text.overlap(channel1, channel1_ref)

## Calculate Average Shift
N0 = np.round(ch1.shape[0]/(1+Noise),0).astype(int)
avg_shift = output_text.avg_dist(ch1[:N0,:].numpy(), ch2[:N0,:].numpy())
avg_shift_map = output_text.avg_dist(ch1[:N0,:].numpy(), ch2_map[:N0,:].numpy())
avg_shift_max = output_text.avg_dist(ch1[:N0,:].numpy(), ch1_ref[:N0,:])


#%% Output

generate_image.plot_channel(channel1, channel2, channel2m, bounds,
                            ref_channel1, precision, reference)
print('Done')

print('\nI: The original overlap was',overlap,'. The mapping has',overlap_map,
      '. The maximum overlap is estimated to be',overlap_max,
      '\nI: The original average distance was',avg_shift,'. The mapping has',avg_shift_map,
      '. The minimum average distance is estimated to be',avg_shift_max)
