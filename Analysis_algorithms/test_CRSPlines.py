# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:48:22 2021

@author: Mels
"""
import sys
sys.path.append('../')

# Packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Classes
from LoadDataModules.Deform import Deform

# Modules
import LoadDataModules.generate_data as generate_data
import MinEntropyModules.Module_Splines as Module_Splines
import MinEntropyModules.Module_ShiftRot as Module_ShiftRot
import OutputModules.output_fn as output_fn


#%% params
Noise=.0
error=.0

deform = Deform(
    deform_on=True,                         # True if we want to give channels deform by hand
    #shift=np.array([ 12  , 9 ]),            # shift in nm
    #rotation=.5,                            # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
    shear=np.array([0.01, 0.011]),         # shear
    scaling=np.array([1.0004,1.0003 ])      # scaling
    )
        
#locs_A, locs_B = generate_data.generate_beads_mimic(deform=deform, N=216, error=error, Noise=Noise)
locs_A, locs_B = generate_data.generate_HEL1_mimic(deform=deform, Nclust=50, error=error, Noise=Noise)

## Splines
gridsize=100
direct=False
nbins = 30                                          # Number of bins

#%% optimizing
ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)
SplinesMod=None
ch2_ShiftRotSpline=None
#'''
# training loop CatmullRomSplines
SplinesMod, ch2_ShiftRotSpline = Module_Splines.run_optimization(ch1, ch2, N_it=200, gridsize=gridsize, 
                                                                 maxDistance=50, learning_rate=1e0,
                                                                 direct=direct, opt=tf.optimizers.Adagrad)
'''
SplinesMod, ch2_ShiftRotSpline = Module_ShiftRot.run_optimization(ch1, ch2, N_it=400, 
                                                                 maxDistance=30, learning_rate=1,
                                                                 direct=direct, opt=tf.optimizers.Adagrad)
'''

print('Optimization Done!')

print('I: Maximum mapping=',np.max( np.sqrt((ch2_ShiftRotSpline[:,0]-ch2[:,0])**2 +
                                            (ch2_ShiftRotSpline[:,1]-ch2[:,1])**2 ) ),'[nm]')

#%% Metrics
plt.close('all')
if True:
    N0 = np.round(ch1.shape[0]/(1+Noise),0).astype(int)
    
    avg1, avg2, fig, (ax1, ax2) = output_fn.errorHist(ch1.numpy(),  ch2.numpy(), ch2_ShiftRotSpline.numpy(), 
                                            nbins=nbins, direct=direct)
    _, _, _, (_, _) = output_fn.errorFOV(ch1.numpy(),  ch2.numpy(), ch2_ShiftRotSpline.numpy(), direct=direct)
    print('\nI: The original average distance was', avg1,'. The mapping has', avg2)
    
    
#%% Plotting the Grid
if SplinesMod is not None:
    output_fn.plot_grid(ch1, ch2, ch2_ShiftRotSpline, SplinesMod, gridsize=gridsize, d_grid = .1, 
                        locs_markersize=4, CP_markersize=8, grid_markersize=3, 
                        grid_opacity=1, lines_per_CP=1)
else:
    print('I: No Spline Mapping used')

print('Done')