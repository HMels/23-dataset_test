# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:32:19 2021

@author: Mels
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

#%% input parameters 
p = Path('dataset_test')
p.mkdir(exist_ok=True)


img_param = setup.Image(zoom = 5, pix_size = 100,            #parameters of the system
                  img_size_x1 = 50, img_size_x2 = 25, 
                  Noise = 40)    

# The data-clusters to be generated
cluster = np.empty(2, dtype = setup.Cluster)
cluster[0] = setup.Cluster(loc_x1 = 5, loc_x2 = 2.5, std_x1 = 2, std_x2 = 4, N = 15)
cluster[1] = setup.Cluster(loc_x1 = 25, loc_x2 = 15, std_x1 = 8, std_x2 = 3, N = 50)

# Deformation of channel B
angle0 = 0.05                         # angle of rotation in degrees
shift0 = np.array([ 2  , 3 ])      # shift in nm

#%% Channel Generation
plt.close('all')

angle = angle0 * np.pi / 180          # angle in radians
shift = shift0 / img_param.zoom      # shift in units of system [zoom]

channel_A, channel_B, localizations_A, localizations_B = (
    setup.run_channel_generation_distribution(cluster, img_param,
                                              angle, shift, error = 0)
    )

#%% Optimizing
ch1 = tf.convert_to_tensor( localizations_A.transpose(), np.float32)
ch2 = tf.convert_to_tensor( localizations_B.transpose(), np.float32)
    
model = Minimum_Entropy.PolMod(name='Polynomial')
opt = tf.optimizers.Adam(learning_rate=0.1)

with tf.GradientTape() as tape:
    y = model(ch1, ch2)
            
gradients = tape.gradient(y, model.trainable_variables)
        
print('- y = ',y.numpy(),'theta = ',model.rotation.theta.numpy(),'shift = ',model.shift.d.numpy())
print('  + gradients = ',gradients)

opt.apply_gradients(zip(gradients, model.trainable_variables))

#%% 

print('- Minimum Entropy:')
print('  + Shift = ',model.shift.d.numpy())
print('  + Rotation = ',model.rotation.theta.numpy())
print('  + y = ',model(ch1, ch2).numpy(),'gradients = ',gradients)