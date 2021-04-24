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
|- setup.py                     File containing everything to setup the program
|  |
|  |- load_data.py              File used to load the example dataset given
|
|- generate_dataset.py          File containing the functions used to generate
|  |                                the localization dataset
|  |- distributions.py          File containing distributions that can be used 
|                                   for the dataset generation
|- dataset_manipulation.py      File containing the functions used to deform
|                                   /manipulate the dataset
|- run_optimization.py          File containing the training loops
|  |
|  |- Minimum_Entropy_Parameterized.py 
|    |                          File containing the Classes and functions necessary
|    |                              for Minimum Entropy 
|    |- ML_functions.py           File containing certain Machine Learning functions
|
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
import setup
import run_optimization
import run_optimization_testversion as run_optimization
import output_text
import generate_image

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)

#%%  The data-clusters to be generated in nm
cluster = np.empty(2, dtype = setup.Cluster)
cluster[0] = setup.Cluster(loc_x1 = 5000, loc_x2 = 4000, std_x1 = 2000, 
                           std_x2 = 4000, N = 5000)
cluster[1] = setup.Cluster(loc_x1 = 20000, loc_x2 = 10000, std_x1 = 7000, 
                           std_x2 = 3000, N = 1500)


#%% Channel Generation
## System Parameters
error = 0.1                                         # localization error in pix
Noise = 0.1                                         # percentage of noise

## Deformation of channel B
angle_degrees = .5                                 # angle of rotation in degrees
shift_nm = np.array([ 40  , 30 ])                  # shift in nm
shear = np.array([0, 0])                            # shear
scaling_rel = np.array([0,0])                       # relative scaling

angle_radians = angle_degrees * np.pi / 180         # angle in radians
scaling = np.array([1, 1]) + scaling_rel            # scaling 

## Dataset
realdata = True
autodeform = True 
# if realdata && autodeform  -> Generate channels via dataset with known deformation
# if realdata && !autodeform -> Generate dataset with unknown everything
# else                       _> Generate channels via distribution   


## Generate Data
localizations_A, localizations_B = setup.run_channel_generation(
    cluster, shift_nm, angle_radians, shear, scaling, error, Noise, 
    realdata, autodeform ) 

ch2 = tf.Variable( localizations_B, dtype = tf.float32)


#%% Minimum Entropy
## Optimization Parameters
Map_options = np.array(['Polynomial', 'Parameterized_simple','Parameterized_complex'])
Map_opt = Map_options[1]                            # what mapping to use
max_deform = 150                                    # maximum amount of deform in nm
learning_rate = 1                                   # learning rate in nm
epochs = 150                                        # amount of iterations of optimization

# Batches used in training 
Batch_on = False
batch_size = 4000                                   # max amount of points per batch
num_batches = np.array([3,3], dtype = int)          # amount of [x1,x2] batches

# Error Message
output_text.Info_batch( np.max([localizations_A.shape[0], localizations_A.shape[0]])
                       , num_batches, batch_size, Batch_on)


## Training Loop
model, loss = run_optimization.initialize_optimizer(
    localizations_A, localizations_B, Map_opt, Batch_on,
    batch_size, num_batches, learning_rate, epochs)


#%% output
ch1, _, ch2_mapped_model = output_text.generate_output(
    localizations_A, localizations_B, model, Map_opt, shift_nm,
    angle_radians, shear, scaling, print_output = (autodeform) and (not realdata) )
        

#%% generating image
## output parameters
precision = 5                                       # precision of image in nm
threshold = 100                                     # threshold for reference points

## Channel Generation
plt.close('all')
channel1, channel2, channel2m, axis = generate_image.generate_channel(
    ch1, ch2, ch2_mapped_model, precision, max_deform)


## Plotting Image
threshold = 100
ref_channel1 = generate_image.reference_clust(ch1, precision * 20, 
                                              axis, threshold)

generate_image.plot_channel(channel1, channel2, channel2m, axis,
                            ref_channel1)

print(loss)
print(model.trainable_variables)