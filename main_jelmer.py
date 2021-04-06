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
|  |- load_data.py              File used to load the example dataset given
|- functions.py                 File containing miscalenious functions
|- generate_dataset.py          File containing the functions used to generate
|  |                                the localization dataset
|  |- distributions.py          File containing distributions that can be used 
|                                   for the dataset generation
|- dataset_manipulation.py      File containing the functions used to deform
|                                   /manipulate the dataset
|- Minimum_Entropy.py           File containing the Classes and functions necessary
|  |                                for the Minimum Entropy Optimization
|  |- ML_functions.py           File containing certain Machine Learning functions


The classes can be found in setup.py
- Cluster()
- Image()

Together with the function:
- run_channel_generation()

It is optional to also run a cross-correlation program
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

# Modules
import cross_correlation
import Minimum_Entropy
import setup

#exec(open("./setup.py").read())
    
#%% input parameters 
p = Path('dataset_test')
p.mkdir(exist_ok=True)


img_param = setup.Image(zoom = 10, pix_size = 100,            #parameters of the system
                  img_size_x1 = 50, img_size_x2 = 25, 
                  Noise = 40)    

# Deformation of channel B
angle0 = 0                          # angle of rotation in degrees
shift0 = np.array([ 0  , 0 ])      # shift in nm

#%% Channel Generation
plt.close('all')

angle = angle0 * np.pi / 180          # angle in radians
shift = shift0 / img_param.zoom      # shift in units of system [zoom]

if True: # generate Channel via real data
    channel_A, channel_B, localizations_A, localizations_B, img_param = (
        setup.run_channel_generation_realdata(img_param,
                                                  angle, shift, error = 0.1)
        )
    
#%% Minimum Entropy
if True: 
    ch1 = tf.convert_to_tensor( localizations_A, np.float32)
    ch2 = tf.convert_to_tensor( localizations_B, np.float32)
    
    polmod = Minimum_Entropy.PolMod(name='Polynomial')
    opt = tf.optimizers.Adam(learning_rate=0.1)
    
    polmod_apply_grads = Minimum_Entropy.get_apply_grad_fn()
    loss = polmod_apply_grads(ch1, ch2, polmod, opt)
    
    print('Minimum Entropy:')
    print(polmod.trainable_weights)
    

    
#%% Run the cross correlation script Monte Carlo Method
#exec(open("./run_cross_cor.py").read())