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
|- setup_image.py               File containing the Deform class and the Image class
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
import generate_data
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
subset = 0.2                                        # percentage of original dataset
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]

## System Parameters
error = 10                                          # localization error in nm
Noise = 0.1                                         # percentage of noise

## Deformation of channel B
shift = np.array([ 40  , 30 ])                      # shift in nm
rotation = .5                                       # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
shear = np.array([0, 0])                            # shear
scaling = np.array([1, 1]) + np.array([0,0])        # scaling 
deform = setup_image.Deform(shift, rotation, shear, scaling)

## Generate Data
locs_A, locs_B = generate_data.run_channel_generation(
    path, deform, error, Noise, realdata, subset
    ) 

ch2 = tf.Variable( locs_B, dtype = tf.float32)


#%% Minimum Entropy
## Optimization Parameters
# decide what mapping to optimize
#model = Minimum_Entropy.ParamMod(name='Parameterized')
model = Minimum_Entropy.Poly2Mod(name='Polynomial')
#model = Minimum_Entropy.Poly3Mod(name='Polynomial')

max_deform = 150                                    # maximum amount of deform in nm
learning_rate = 1e-7                                        # learning rate 
epochs = 300                                        # amount of iterations of optimization

# Batches used in training 
Batch_on = False
batch_size = 4000                                   # max amount of points per batch
num_batches = np.array([3,3], dtype = int)          # amount of [x1,x2] batches

# Error Message
output_text.Info_batch( np.max([locs_A.shape[0], locs_B.shape[0]])
                       , num_batches, batch_size, Batch_on)


## Training Loop
model, loss = run_optimization.run_optimization(
    locs_A, locs_B, model, Batch_on,
    batch_size, num_batches, learning_rate, epochs)


#%% output
ch1, _, ch2_mapped_model = output_text.generate_output(
    locs_A, locs_B, model, deform)
        

#%% generating image
## output parameters
precision = 5                                       # precision of image in nm
threshold = 100                                     # threshold for reference points

## Channel Generation
plt.close('all')
channel1, channel2, channel2m, axis = generate_image.generate_channel(
    ch1, ch2, ch2_mapped_model, precision, max_deform)

#%%
## Plotting Image
ref_channel1 = generate_image.reference_clust(ch1, precision * 20, 
                                              axis, threshold)

generate_image.plot_channel(channel1, channel2, channel2m, axis,
                            ref_channel1)

print('\n\n')
print(loss)
print(model.trainable_variables)