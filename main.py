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
from LoadDataModules.Deform import Deform

# Modules
import LoadDataModules.generate_data as generate_data
import OutputModules.output_fn as output_fn
import OutputModules.generate_image as generate_image

# Models
import MinEntropyModules.Module_ShiftRot as Module_ShiftRot
import MinEntropyModules.Module_Splines as Module_Splines
import MinEntropyModules.Module_Poly3 as Module_Poly3

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)


#%% Channel Generation
## Dataset
coupled = True                               # True if data is coupled 
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]

## System Parameters
error = 0.0                                 # localization error in nm
Noise = 0.0                                 # percentage of noise

## Channel B
copy_channel=False
deform = Deform(
    deform_on=False,                         # True if we want to give channels deform by hand
    shift=np.array([ 12  , 9 ]),            # shift in nm
    rotation=.5,                            # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
    #shear=np.array([0.003, 0.002]),         # shear
    #scaling=np.array([1.0004,1.0003 ])      # scaling
    )


locs_A, locs_B = generate_data.generate_channels(
    path=path, deform=deform, error=error, Noise=Noise, copy_channel=copy_channel,
    subset=1,                               # the subset of the dataset we want to load
    pix_size=1                              # size of a pixel in nm
    )

#locs_A, locs_B = generate_data.generate_channels_random(216, deform, error=error, Noise=Noise)


#%% Minimum Entropy
## Params
N_it = [500, 500]                                 # The number of iterations in the training loop
gridsize = 200                                      # The size of the grid of the Splines

## In tf format
ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)

# Error Message
output_fn.Info_batch( locs_A.shape[0], locs_B.shape[0], coupled)

# Initialize used variables
ShiftRotMod=None
ch2_ShiftRot=None
SplinesMod=None
ch2_ShiftRotSpline=None


#%% ShiftRotMod
# training loop ShiftRotMod
ShiftRotMod, ch2_ShiftRot = Module_ShiftRot.run_optimization(ch1, ch2, N_it=N_it[0], maxDistance=30, 
                                                            threshold=10, learning_rate=1,
                                                            direct=coupled)

if ShiftRotMod is not None: 
    print('I: Shift Mapping=', ShiftRotMod.model.trainable_variables[0].numpy(), 'nm')
    print('I: Rotation Mapping=', ShiftRotMod.model.trainable_variables[1].numpy()/100,'degrees')
else:
    print('I: No shift or rotation mapping used')


#%% Splines
# training loop CatmullRomSplines
SplinesMod, ch2_ShiftRotSpline = Module_Splines.run_optimization(ch1, ch2_ShiftRot, N_it=N_it[1],
                                                                 gridsize=gridsize, threshold=10,
                                                                 maxDistance=30, learning_rate=1e-3, 
                                                                 direct=coupled)



print('Optimization Done!')
if ch2_ShiftRotSpline is not None:
    print('I: Maximum mapping=',np.max( np.sqrt((ch2_ShiftRotSpline[:,0]-ch2[:,0])**2 +
                                     (ch2_ShiftRotSpline[:,1]-ch2[:,1])**2 ) ),'[nm]')
else:
    print('I: Maximum mapping=',np.max( np.sqrt((ch2_ShiftRot[:,0]-ch2[:,0])**2 +
                                     (ch2_ShiftRot[:,1]-ch2[:,1])**2 ) ),'[nm]')
    

#%% Metrics
# Histogram
hist_output = True                                  # do we want to have the histogram output
nbins = 30                                          # Number of bins

plt.close('all')
if hist_output:
    N0 = np.round(ch1.shape[0]/(1+Noise),0).astype(int)
    
    avg1, avg2 = output_fn.errorHist(ch1[:N0,:].numpy(),  ch2[:N0,:].numpy(),
                                            #ch2_ShiftRotSpline[:N0,:].numpy(), 
                                            ch2_ShiftRot[:N0,:].numpy(),
                                            nbins=nbins, direct=coupled)
    _, _ = output_fn.errorFOV(ch1[:N0,:].numpy(),  ch2[:N0,:].numpy(), 
                              #ch2_ShiftRotSpline[:N0,:].numpy(),
                              ch2_ShiftRot[:N0,:].numpy(),
                              direct=coupled)
    print('\nI: The original average distance was', avg1,'. The mapping has', avg2)


#%% generating image
# The Image
plot_img = True                                     # do we want to generate a plot
reference = False                                   # do we want to plot reference points
precision = 5                                       # precision of image in nm
threshold = 100                                     # threshold for reference points

if plot_img:
    ## Channel Generation
    channel1, channel2, channel2m, bounds = generate_image.generate_channel(
        ch1, ch2,
        ch2_ShiftRot,
        #ch2_ShiftRotSpline,
        precision)
    
    
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
    
    generate_image.plot_1channel(channel1, bounds, ref_channel1, precision, reference=False)


#%% Plotting the Grid
if SplinesMod is not None:
    output_fn.plot_grid(ch1, ch2_ShiftRot, ch2_ShiftRotSpline, SplinesMod, gridsize=gridsize, d_grid = .2, 
                        locs_markersize=10, CP_markersize=8, grid_markersize=3, 
                        grid_opacity=1, lines_per_CP=1)
else:
    print('I: No Spline Mapping used')

print('Done')