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
from setup_image import Deform

# Modules
import generate_data
from run_optimization import Models
#import MinEntropy
import MinEntropy_direct as MinEntropy

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)



#%% The opt and errorbar algorithm
def run_errorbar_algorithm(locs, Mod, Noise=0., realdata=False, 
                           shift=np.array([ 17  , 19 ])):
    locs_A=locs[0]
    locs_B=locs[1]
    
    locs_B[:,0]-=shift[0]
    locs_B[:,1]-=shift[1]
    
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)

    # training loop
    mods=Models(model=Mod[0][0], learning_rate=Mod[1][0], 
                            opt=Mod[2][0] )
    
    i=0 
    while not mods.endloop:
        
        mods.Training_loop(ch1, ch2)                         # the training loop        
        i+=1     
                      
    print('completed in',i,' iterations')
    print('Model: ', mods.model)
    print('+ variables',mods.var)
    print('\n')
    
    if realdata: N0 = ch1.shape[0]
    else: N0 = np.round(ch1.shape[0]/(1+Noise),0).astype(int)
    mapping = mods.model.trainable_variables[0].numpy()
    
    del locs_A, locs_B,ch1, ch2, mods,i
    return mapping, N0


#%% Error calc
N=10
Nit=1
shift = np.array([ np.linspace(0,10,N), np.zeros(N)])
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]
subset = 1
pixsize=1

N0 = np.zeros(N)
mapping = np.zeros([N,2])

Def =  Deform()
locs_A , _ = generate_data.run_channel_generation(
            path, Def, error=0, Noise=0, realdata=False, subset=subset, pix_size=pixsize
            )

for i in range(N):
    print(' -------------------- iteration:',i,'--------------------')
    for j in range(Nit):
        models = [MinEntropy.ShiftMod()]
        optimizers = [tf.optimizers.Adagrad]
        learning_rates = np.array([.1])
        Mods = [models, learning_rates, optimizers]
        
        locs = [locs_A.copy(), locs_A.copy()]
        
        mapping_temp, N0_temp = run_errorbar_algorithm(locs, Mods, shift=shift[:,i])
        mapping[i,:]+=mapping_temp/Nit
        N0[i]+=N0_temp/Nit
        
        del Mods, models, optimizers, learning_rates, locs
    

#%%
plt.close('all')
plt.title(('Beads dataset with ',locs_A.shape[0],'localizations'))
plt.plot(shift[0,:], mapping[:,0],'ro',ls='',label='Calculated Shift')
plt.plot(shift[0,:], np.abs(mapping[:,0]-shift[0,:]),'bx',label='Error', ls='')
plt.plot(shift[0,:],shift[0,:],label='Actual Shift')
plt.legend()
plt.xlabel('Original Shift [nm]')
plt.ylabel('Calculated Shift [nm]')
plt.xlim([0,np.max(shift[0,:])])
plt.ylim([0,np.max([np.max(shift[0,:]),np.max(mapping[:,0])])])
plt.show()

#%% Error calc
N=40
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
subset = np.linspace(.1,1,N)

N0 = np.zeros(N)
mapping = np.zeros([N,2])
shift=np.array([ 7  , 9 ])

Def =  Deform()
for i in range(N):
    print(' -------------------- iteration:',i,'--------------------')    

    locs_A , _ = generate_data.run_channel_generation(
        path, Def, error=0, Noise=0, realdata=False, subset=subset[i], pix_size=100
        )
    
    models = [MinEntropy.ShiftMod()]
    optimizers = [tf.optimizers.Adagrad]
    learning_rates = np.array([1.])
    Mods = [models, learning_rates, optimizers]
    
    locs = [locs_A.copy(), locs_A.copy()]
    
    mapping[i,:], N0[i] = run_errorbar_algorithm(locs, Mods, shift=shift)
    
    del Mods, models, optimizers, learning_rates, locs, locs_A


#%%
plt.figure()
plt.title('Error vs Number of Localizations')
plt.plot(N0, np.sqrt(np.sum( (mapping-shift)**2 , axis = 1)),'ro',ls='')
plt.xlabel('N')
plt.ylabel('Error [nm]')
plt.xlim(0)
plt.ylim(0)
plt.show()
