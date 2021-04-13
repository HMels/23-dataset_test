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
import Minimum_Entropy
import setup
import dataset_manipulation

#exec(open("./setup.py").read())
#%reload_ext tensorboard

#%% input parameters 
p = Path('dataset_test')
p.mkdir(exist_ok=True)

# The data-clusters to be generated
cluster = np.empty(2, dtype = setup.Cluster)
cluster[0] = setup.Cluster(loc_x1 = 5, loc_x2 = 2.5, std_x1 = 2, std_x2 = 4, N = 350)
cluster[1] = setup.Cluster(loc_x1 = 25, loc_x2 = 15, std_x1 = 8, std_x2 = 3, N = 1000)

# Deformation of channel B
angle_degrees = .5                              # angle of rotation in degrees
angle_radians = angle_degrees * np.pi / 180     # angle in radians
shift_nm = np.array([ 10  , 15 ])               # shift in nm

#%% Channel Generation
if True: # generate Channel via distribution
    localizations_A, localizations_B = (
        setup.run_channel_generation_distribution(cluster, angle_radians, shift_nm, 
                                                  error = 0.1, Noise = 0.1)
        )

if False: # generate Channel via real data
    localizations_A, localizations_B = (
        setup.run_channel_generation_realdata(angle_radians, shift_nm, error = 0.1,
                                              batch_size = 0.01, Noise = 0.1)
        )
    
#%% Minimum Entropy
if True:
    ch1 = tf.Variable( localizations_A, dtype = tf.float32)
    ch2 = tf.Variable( localizations_B, dtype = tf.float32)
        
    ch2_mapped = tf.Variable(
        dataset_manipulation.rotation( dataset_manipulation.shift(
                localizations_B, -1 * shift_nm) , -1 * angle_radians) , 
        dtype = tf.float32)
    
    ##########################################################################
    print('\n-------------------- TARGET -------------------------')
    print('+ Shift = ', shift_nm, ' [nm]')
    print('+ Rotation = ', angle_radians, ' [radians]')
    print('+ Entropy = ', Minimum_Entropy.Rel_entropy(ch1, ch2_mapped ).numpy())
    ##########################################################################
    
    model = Minimum_Entropy.PolMod(name='Polynomial')
    opt = tf.optimizers.Adam(learning_rate=.1)
    
    model_apply_grads = Minimum_Entropy.get_apply_grad_fn()
    loss = model_apply_grads(ch1, ch2, model, opt, error = .05)
    
    ##########################################################################
    print('\n-------------------- RESULT --------------------------')
    print('+ Shift = ',model.shift.d.numpy(), ' [nm]')
    print('+ Rotation = ',model.rotation.theta.numpy()*180/np.pi,' [degrees]')
    print('+ Entropy = ',model(ch1, ch2).numpy())
    
    print('\n-------------------- COMPARISSON ---------------------')
    print('+ Shift = ', shift_nm, ' [nm]')
    print('+ Rotation = ', angle_degrees, ' [degrees]')
    print('+ Entropy = ', Minimum_Entropy.Rel_entropy(ch1, ch2_mapped ).numpy())
    ##########################################################################
    

#%% plotting
plt.close('all')
plt.plot( localizations_A[:,0], localizations_A[:,1], 'ro', ls = '')
    
#%% Run the cross correlation script Monte Carlo Method
#exec(open("./run_cross_cor.py").read())