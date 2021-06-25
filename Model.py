# Model.py
'''
_______________________________________________________________________________

WARNING: CHANGING THE MODEL MEANS YOU ALSO HAVE TO CHANGE transform_vec()
_______________________________________________________________________________
'''
import tensorflow as tf
import numpy as np
from copy import copy
import time


# Modules
import OutputModules.output_fn as output_fn


# Models
import MinEntropyModules.Module_ShiftRot as Module_ShiftRot
import MinEntropyModules.Module_Splines as Module_Splines
import MinEntropyModules.Module_Poly3 as Module_Poly3


#%% run_model
def run_model(ch1, ch2, coupled=True, N_it=[400, 200], learning_rate=[1,1e-2], 
              gridsize = 100, plot_grid=False, sys_param=None):
    '''
    

    Parameters
    ----------
    ch1 , ch2 : tensor
        Tensors containing the [x1,x2] locations of the original channels.
    coupled : bool, optional
        True if the dataset is coupled. The default is True.
    N_it : List, optional
        List containing how many iterations we train each model. The default is [400, 200].
    learning_rate : List, optional
        List containing the learning rates used per model. The default is [1,1e-2].
    gridsize : int, optional
        Gridsize of the Splines in nm. The default is 100.
    plot_grid : bool, optional
        True if we want to plot the Spline grid. The default is False.
    sys_params : list, optional
        List containing the size of the system. The optional is None,
        which means it will be calculated by hand

    Returns
    -------
    mods : list
        List containing the models used.
    ch2_mapped : tensor
        Tensor containing the [x1,x2] locations of the mapped dataset.

    '''        
    # Error Message
    output_fn.Info_batch( ch1.shape[0], ch2.shape[0], coupled)
    
    # Initialize used variables
    start = time.time()
    ShiftRotMod=None
    ch2_ShiftRot=None
    SplinesMod=None
    ch2_ShiftRotSpline=None
    
    
    #% ShiftRotMod
    # training loop ShiftRotMod
    print('Running ShiftRot model...')
    ShiftRotMod, ch2_ShiftRot = Module_ShiftRot.run_optimization(
        ch1, ch2, N_it=N_it[0], maxDistance=30, learning_rate=learning_rate[0],
        direct=coupled, opt=tf.optimizers.Adagrad
        )
    
    print('I: Shift Mapping=', ShiftRotMod.model.trainable_variables[0].numpy(), 'nm')
    print('I: Rotation Mapping=', ShiftRotMod.model.trainable_variables[1].numpy()/100,'degrees')
    
    
    #% Splines
    # training loop CatmullRomSplines
    print('Running Splines model...')
    SplinesMod, ch2_ShiftRotSpline = Module_Splines.run_optimization(
        ch1, ch2_ShiftRot, N_it=N_it[1], gridsize=gridsize, maxDistance=30, 
        learning_rate=learning_rate[1], direct=coupled,  opt=tf.optimizers.Adagrad,
        sys_param=sys_param
        )
    
    print('I: Maximum mapping=',np.max( np.sqrt((ch2_ShiftRotSpline[:,0]-ch2[:,0])**2 +
                                         (ch2_ShiftRotSpline[:,1]-ch2[:,1])**2 ) ),'[nm]')
       
    
        
    #% Plotting the Grid
    if SplinesMod is not None and plot_grid:
        output_fn.plot_grid(ch1, ch2_ShiftRot, ch2_ShiftRotSpline, SplinesMod, gridsize=gridsize, d_grid = .01, 
                            locs_markersize=10, CP_markersize=8, grid_markersize=3, 
                            grid_opacity=1, lines_per_CP=4, sys_param=sys_param)
        
    print('Optimization Done in ',round(time.time()-start),'seconds')
    return [ShiftRotMod, SplinesMod], ch2_ShiftRotSpline
        

#%% transform_vec
def transform_vec(mods, ch2, gridsize, sys_param=None):
    '''
    Transforms the vector according to the model

    Parameters
    ----------
    mods : list
        list containing the model.
    ch2 : tf.Tensor
        the original channel 2.
    gridsize : float
        Size of the grid in nm.
    sys_params : list, optional
        List containing the size of the system. The optional is None,
        which means it will be calculated by hand

    Returns
    -------
    ch2_transformed : tf.Tensor
        transformed ch2.

    '''
    
    ## Shiftrot  
    # transform ch2 with ShiftRot
    ch2_transformed = mods[0].model.transform_vec(ch2)
    
    
    ## Splines
    # Creating the right control points for the Splines
    ch2_transformed = tf.Variable(ch2_transformed/gridsize, trainable=False)
    if sys_param is None:
        x1_min = tf.reduce_min(tf.floor(ch2_transformed[:,0]))
        x2_min = tf.reduce_min(tf.floor(ch2_transformed[:,1]))
    else:
        x1_min = tf.floor(sys_param[0,0]/gridsize)
        x2_min = tf.floor(sys_param[0,1]/gridsize)
        
    CP_idx = tf.cast(tf.stack(
        [( ch2_transformed[:,0]-x1_min+1)//1 , ( ch2_transformed[:,1]-x2_min+1)//1 ], 
        axis=1), dtype=tf.int32)
    
    # transform ch2 with Splines
    mods_temp = copy(mods[1].model)
    mods_temp.reset_CP(CP_idx)
    ch2_transformed = mods_temp.transform_vec(ch2_transformed)
    return ch2_transformed*gridsize
