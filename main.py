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
import Model
import Cross_validate


#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)

plt.close('all')


#%% Channel Generation
## Dataset
dataset=[ # [ Path, pix_size, coupled, spline gridsize, subset ]
    [ [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ], 1, True, 100, 1 ],
    [ [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
     'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ], 100, False, 1000, .4 ]
      ]
ds = dataset[0]      # [0] for beads [1] for HEL1


## Deformation and system parameters
error = 0.0                                             # localization error in nm
Noise = 0.0   
deform = Deform(deform_on=False,                        # True if we want to give channels deform by hand
                shift=np.array([ 12  , 9 ]),            # shift in nm 
                rotation=.5,                            # angle of rotation in degrees
                shear=np.array([0.003, 0.002]),         # shear
                scaling=np.array([1.0004,1.0003 ]),     # scaling
                random_deform=False                     # True if we want to randomly generate deform
                )
copy_channel = False
    

## Load Data
locs_A, locs_B = generate_data.generate_channels(
    path=ds[0], deform=deform, error=.0, Noise=.0,
    copy_channel=copy_channel, subset=ds[4], pix_size=ds[1])


#%% Minimum Entropy Model
## Cross Reference
avg_trained, avg_crossref = Cross_validate.cross_validate(locs_A, locs_B, coupled=ds[2], gridsize=ds[3],
                                                      plot_hist=False, plot_FOV=False)

print('Trained Error=',avg_trained,'nm. Cross Reference Error=',avg_crossref,'nm')
#input("Press Enter to continue...")


#%% Run Model
ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)
mods, ch2_mapped = Model.run_model(ch1, ch2, coupled=ds[2], N_it=[400, 200], 
                                   learning_rate=[1,1e-2], gridsize = ds[3],
                                   plot_grid=False)


#%% Metrics
## Histogram
nbins = 30                                          # Number of bins
avg1, avg2, fig1, _ = output_fn.errorHist(ch1,  ch2, ch2_mapped, nbins=nbins, direct=ds[2])
fig1.suptitle('Distribution of distances between neighbouring Localizations')
    

## FOV
_, _, fig2, _ = output_fn.errorFOV(ch1,  ch2, ch2_mapped, direct=ds[2])
fig2.suptitle('Distribution of error between neighbouring pairs over radius')
    
print('\nI: The original average distance was', avg1,'. The mapping has', avg2)


#%% generating image
# The Image
plot_img = True                                     # do we want to generate a plot
precision = 5                                       # precision of image in nm
threshold = 100                                     # threshold for reference points

if plot_img:
    ## Channel Generation
    channel1, channel2, channel2m, bounds = generate_image.generate_channel(
        ch1, ch2, ch2_mapped, precision)

    ## estimate original ch2
    ch1_ref = generate_data.localization_error(locs_A, error)
    channel1_ref = generate_image.generate_matrix(ch1_ref / precision, bounds)
    
    
    ## Plotting
    generate_image.plot_channel(channel1, channel2, channel2m, bounds, precision)
    #generate_image.plot_1channel(channel1, bounds, precision)


print('\n Optimization Done')