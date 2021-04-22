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
import matplotlib.pyplot as plt
from pathlib import Path

# Modules
import setup
import run_optimization
import output_text
import generate_image

#exec(open("./setup.py").read())
#%reload_ext tensorboard

plt.close('all')
p = Path('dataset_test')
p.mkdir(exist_ok=True)

#%% input parameters 
pix_size = 100

# The data-clusters to be generated in pix
cluster = np.empty(2, dtype = setup.Cluster)
cluster[0] = setup.Cluster(loc_x1 = 50, loc_x2 = 40, std_x1 = 20, 
                           std_x2 = 40, N = 5000)
cluster[1] = setup.Cluster(loc_x1 = 200, loc_x2 = 100, std_x1 = 70, 
                           std_x2 = 30, N = 1500)

## System Parameters
error = 0.1                                         # the localization error in pix
Noise = 0.1                                         # the percentage of noise

# Deformation of channel B
angle_degrees = 0.5                                 # angle of rotation in degrees
shift_nm = np.array([ 80  , 120 ])                  # shift in nm
shear = np.array([0, 0])                            # shear
scaling_rel = np.array([0,0])                       # amount of relative scaling

shift_pix = shift_nm / pix_size                     # shift in pix
angle_radians = angle_degrees * np.pi / 180         # angle in radians
scaling = np.array([1, 1]) + scaling_rel            # scaling 

# Mapping Function
Map_options = np.array(['Polynomial', 'Parameterized_simple','Parameterized_complex'])
Map_opt = Map_options[1]

# Batches used in training 
Batch_on = True
batch_size = 6000
num_batches = np.array([3,3], dtype = int)

#%% Channel Generation
realdata = True
autodeform = False 
# if realdata && autodeform  -> Generate channels via dataset with known deformation
# if realdata && !autodeform -> Generate dataset with unknown everything
# else                       _> Generate channels via distribution   
localizations_A, localizations_B = setup.run_channel_generation(
    cluster, shift_pix, angle_radians, shear, scaling, error, Noise, 
    realdata, autodeform ) 


output_text.Info_batch(localizations_A.shape[0], num_batches, batch_size, Batch_on)

#%% Minimum Entropy
model, loss = run_optimization.initialize_optimizer(
    localizations_A, localizations_B, Map_opt, Batch_on,
    batch_size, num_batches, learning_rate=.05)


#%% output
if (autodeform) and (not realdata):
    ch1, ch2, ch2_mapped_model = output_text.generate_output(
        localizations_A, localizations_B, model, pix_size, Map_opt, shift_pix,
        angle_radians, shear, scaling )
        

#%%
import generate_image

precision = 5 # precision of image in nm

plt.close('all')
generate_image.plot_channel(ch1, ch2, ch2_mapped_model, precision, pix_size)