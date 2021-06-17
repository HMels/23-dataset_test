# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:48:22 2021

@author: Mels
"""
# Packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Classes
from LoadDataModules.Deform import Deform

# Modules
import LoadDataModules.generate_data as generate_data
import MinEntropyModules.Module_Splines as Module_Splines
import OutputModules.output_fn as output_fn
import MinEntropyModules.train_model as train_model

'''
_______________________________________________________________________________

Obiously the CatmullRomSplines model described here (the one with free borders)
does not work for a bit. The one described by Module_Splines works just fine! 
I will continue to work on the different run___algorithm.py files and test the 
Splines on them
_______________________________________________________________________________
'''


#%% CatmullRomSplines
class CatmullRomSplines(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, CP_locs, CP_idx, ch2, CP_idx_nn=None, name='CatmullRomSplines'):
        super().__init__(name=name)

        # The location of the ControlPoints. This will be trained
        self.CP_locs = tf.Variable(CP_locs, dtype=tf.float32,
                                   trainable=True, name='ControlPointstrainable')  
        
        # The indices of which locs in ch2 belong to which CP_locs
        self.CP_idx = tf.Variable(CP_idx, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
        self.CP_idx_nn = (tf.Variable(CP_idx_nn, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
                          if CP_idx_nn is not None else {})
        
        self.A = tf.Variable([
            [-.5, 1.5, -1.5, 0.5],
            [1, -2.5, 2, -.5],
            [-.5, 0, .5, 0],
            [0, 1, 0, 0]
            ], trainable=False, dtype=tf.float32)
        self.r = tf.Variable(ch2%1, trainable=False, dtype=tf.float32, 
                             name='Distance to ControlPoinst')
        

    
    @tf.function 
    def call(self, ch1, ch2):
        ch2_mapped = self.transform_vec(ch2)
        return tf.reduce_sum(ch2_mapped**2-ch1**2)
    
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        self.update_splines(self.CP_idx)        
        x = x_input[:,0][:,None]%1
        y = x_input[:,1][:,None]%1
        
        M_matrix = tf.stack([
            tf.pow(x,3)*tf.pow(y,3)*self.Sum_A(0,0),
            tf.pow(x,3)*tf.pow(y,2)*self.Sum_A(0,1),
            tf.pow(x,3)*tf.pow(y,1)*self.Sum_A(0,2),
            tf.pow(x,3)*tf.pow(y,0)*self.Sum_A(0,3),
            
            tf.pow(x,2)*tf.pow(y,3)*self.Sum_A(1,0),
            tf.pow(x,2)*tf.pow(y,2)*self.Sum_A(1,1),
            tf.pow(x,2)*tf.pow(y,1)*self.Sum_A(1,2),
            tf.pow(x,2)*tf.pow(y,0)*self.Sum_A(1,3),
        
            tf.pow(x,1)*tf.pow(y,3)*self.Sum_A(2,0),
            tf.pow(x,1)*tf.pow(y,2)*self.Sum_A(2,1),
            tf.pow(x,1)*tf.pow(y,1)*self.Sum_A(2,2),
            tf.pow(x,1)*tf.pow(y,0)*self.Sum_A(2,3),
            
            tf.pow(x,0)*tf.pow(y,3)*self.Sum_A(3,0),
            tf.pow(x,0)*tf.pow(y,2)*self.Sum_A(3,1),
            tf.pow(x,0)*tf.pow(y,1)*self.Sum_A(3,2),
            tf.pow(x,0)*tf.pow(y,0)*self.Sum_A(3,3),
            ], axis=2)
        return tf.reduce_sum(M_matrix, axis=2)
    
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def transform_mat(self, x_input):
        self.load_CPlocs()
        self.update_splines(self.CP_idx_nn)        
        x = x_input[:,:,0][:,:,None]%1
        y = x_input[:,:,1][:,:,None]%1
        
        M_matrix = tf.stack([
            tf.pow(x,3)*tf.pow(y,3)*self.Sum_A(0,0),
            tf.pow(x,3)*tf.pow(y,2)*self.Sum_A(0,1),
            tf.pow(x,3)*tf.pow(y,1)*self.Sum_A(0,2),
            tf.pow(x,3)*tf.pow(y,0)*self.Sum_A(0,3),
            
            tf.pow(x,2)*tf.pow(y,3)*self.Sum_A(1,0),
            tf.pow(x,2)*tf.pow(y,2)*self.Sum_A(1,1),
            tf.pow(x,2)*tf.pow(y,1)*self.Sum_A(1,2),
            tf.pow(x,2)*tf.pow(y,0)*self.Sum_A(1,3),
        
            tf.pow(x,1)*tf.pow(y,3)*self.Sum_A(2,0),
            tf.pow(x,1)*tf.pow(y,2)*self.Sum_A(2,1),
            tf.pow(x,1)*tf.pow(y,1)*self.Sum_A(2,2),
            tf.pow(x,1)*tf.pow(y,0)*self.Sum_A(2,3),
            
            tf.pow(x,0)*tf.pow(y,3)*self.Sum_A(3,0),
            tf.pow(x,0)*tf.pow(y,2)*self.Sum_A(3,1),
            tf.pow(x,0)*tf.pow(y,1)*self.Sum_A(3,2),
            tf.pow(x,0)*tf.pow(y,0)*self.Sum_A(3,3),
            ], axis=3)
        return tf.reduce_sum(M_matrix, axis=3)
        
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def Sum_A(self,a,b):
        A_matrix = tf.stack([
            self.A[a,0]*self.A[b,0]*self.q00,
            self.A[a,0]*self.A[b,1]*self.q01,
            self.A[a,0]*self.A[b,2]*self.q02,
            self.A[a,0]*self.A[b,3]*self.q03,
            
            self.A[a,1]*self.A[b,0]*self.q10,
            self.A[a,1]*self.A[b,1]*self.q11,
            self.A[a,1]*self.A[b,2]*self.q12,
            self.A[a,1]*self.A[b,3]*self.q13,
            
            self.A[a,2]*self.A[b,0]*self.q20,
            self.A[a,2]*self.A[b,1]*self.q21,
            self.A[a,2]*self.A[b,2]*self.q22,
            self.A[a,2]*self.A[b,3]*self.q23,
            
            self.A[a,3]*self.A[b,0]*self.q30,
            self.A[a,3]*self.A[b,1]*self.q31,
            self.A[a,3]*self.A[b,2]*self.q32,
            self.A[a,3]*self.A[b,3]*self.q33
            ], axis=2)
        return tf.reduce_sum(A_matrix, axis=2)
        
    
    #@tf.function
    def update_splines(self, idx):
        self.q00 = tf.gather_nd(self.CP_locs, idx+[-1,-1])  # q_k
        self.q01 = tf.gather_nd(self.CP_locs, idx+[-1,0])  # q_k
        self.q02 = tf.gather_nd(self.CP_locs, idx+[-1,1])  # q_k
        self.q03 = tf.gather_nd(self.CP_locs, idx+[-1,2])  # q_k
            
        self.q10 = tf.gather_nd(self.CP_locs, idx+[0,-1])  # q_k
        self.q11 = tf.gather_nd(self.CP_locs, idx+[0,0])  # q_k
        self.q12 = tf.gather_nd(self.CP_locs, idx+[0,1])  # q_k
        self.q13 = tf.gather_nd(self.CP_locs, idx+[0,2])  # q_k
            
        self.q20 = tf.gather_nd(self.CP_locs, idx+[1,-1])  # q_k
        self.q21 = tf.gather_nd(self.CP_locs, idx+[1,0])  # q_k
        self.q22 = tf.gather_nd(self.CP_locs, idx+[1,1])  # q_k
        self.q23 = tf.gather_nd(self.CP_locs, idx+[1,2])  # q_k
            
        self.q30 = tf.gather_nd(self.CP_locs, idx+[2,-1])  # q_k
        self.q31 = tf.gather_nd(self.CP_locs, idx+[2,0])  # q_k
        self.q32 = tf.gather_nd(self.CP_locs, idx+[2,1])  # q_k
        self.q33 = tf.gather_nd(self.CP_locs, idx+[2,2])  # q_k
        
        
    def reset_CP(self, CP_idx, CP_idx_nn=None):
        # The indices of which locs in ch2 belong to which CP_locs
        self.CP_idx = tf.Variable(CP_idx, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
        self.CP_idx_nn = (tf.Variable(CP_idx_nn, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
                          if CP_idx_nn is not None else {})



#%% run_optimization
def run_optimization(ch1, ch2, N_it=3000, gridsize=50, threshold=10, maxDistance=50, 
                             learning_rate=1e-3, direct=False):
    '''
    Parameters
    ----------
    ch1 ,ch2 : Nx2 Tensor
        Tensor containing the localizations of their channels.
    gridsize : float32, optional
        The size of the grid splines
    threshold : int, optional
        The amount of rejections before ending the optimization loop.
        The default is 10.
    maxDistance : float32, optional
        The distance in which the Nearest Neighbours will be searched. 
        The default is 50nm.
    learning_rate : float, optional
        The initial learning rate of our optimizer. the default is 1.
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.

    Returns
    -------
    mods : Models() Class
        Class containing the information and functions of the optimization models.
    ch2 : Nx2 float Tensor
        Mapped version of ch2

    '''   
    ch1_input = tf.Variable(ch1/gridsize, trainable=False)
    ch2_input = tf.Variable(ch2/gridsize, trainable=False)

        
    CP_idx = tf.cast(tf.stack(
        [( ch2_input[:,0]-tf.reduce_min(tf.floor(ch2_input[:,0]))+1)//1 , 
         ( ch2_input[:,1]-tf.reduce_min(tf.floor(ch2_input[:,1]))+1)//1 ], 
        axis=1), dtype=tf.int32)
        
    if True:          # direct 
        nn1=None
        nn2=None
        
        x1_grid = tf.range(tf.reduce_min(tf.floor(ch2_input[:,0])) -1,
                       tf.reduce_max(tf.floor(ch2_input[:,0])) +3, 1)
        x2_grid =  tf.range(tf.reduce_min(tf.floor(ch2_input[:,1]))-1,
                            tf.reduce_max(tf.floor(ch2_input[:,1])) +3, 1)
        CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
        model = CatmullRomSplines(CP_locs, CP_idx, ch2_input)
    
    # The Model
    mods = train_model.Models(model=model, learning_rate=learning_rate, 
                  opt=tf.optimizers.Adagrad, threshold=threshold)
    
    # The Training Function
    model_apply_grads = train_model.get_apply_grad_fn()
    mods, ch2_input = model_apply_grads(ch1=ch1_input, ch2=ch2_input, N_it=N_it,
                                        mods=mods, nn1=None, nn2=None)
    return mods, ch2_input*gridsize


#%% params
Noise=.0
error=.0

deform = Deform(
    deform_on=True,                         # True if we want to give channels deform by hand
    #shift=np.array([ 12  , 9 ]),            # shift in nm
    #rotation=.5,                            # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
    shear=np.array([0.003, 0.002]),         # shear
    scaling=np.array([1.0004,1.0003 ])      # scaling
    )
        
#locs_A, locs_B = generate_data.generate_beads_mimic(Nlocs[i], deform, error=error, Noise=Noise)
locs_A, locs_B = generate_data.generate_HEL1_mimic(Nclust=10,deform=deform, error=error, Noise=Noise)

## Splines
gridsize=1000
direct=True
bin_width=20

#%% optimizing
ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)
SplinesMod=None
ch2_ShiftRotSpline=None

# training loop CatmullRomSplines
#'''
SplinesMod, ch2_ShiftRotSpline = run_optimization(ch1, ch2, N_it=500, gridsize=gridsize, 
                                                  maxDistance=30, learning_rate=1e-4,
                                                  direct=direct)
'''
SplinesMod, ch2_ShiftRotSpline = Module_Splines.run_optimization(ch1, ch2, N_it=500, gridsize=gridsize, 
                                                                 maxDistance=30, learning_rate=1e-2,
                                                                 direct=direct)
'''

print('Optimization Done!')

print('I: Maximum mapping=',np.max( np.sqrt((ch2_ShiftRotSpline[:,0]-ch2[:,0])**2 +
                                            (ch2_ShiftRotSpline[:,1]-ch2[:,1])**2 ) ),'[nm]')

#%% Metrics
plt.close('all')
if True:
    N0 = np.round(ch1.shape[0]/(1+Noise),0).astype(int)
    
    avg1, avg2 = output_fn.errorHist(ch1[:N0,:].numpy(),  ch2[:N0,:].numpy(),
                                            ch2_ShiftRotSpline[:N0,:].numpy(), 
                                            bin_width, direct=direct)
    _, _ = output_fn.errorFOV(ch1[:N0,:].numpy(),  ch2[:N0,:].numpy(), 
                              ch2_ShiftRotSpline[:N0,:].numpy(),
                              direct=direct)
    print('\nI: The original average distance was', avg1,'. The mapping has', avg2)
    
    
#%% Plotting the Grid
if SplinesMod is not None:
    output_fn.plot_grid(ch1, ch2, ch2_ShiftRotSpline, SplinesMod, gridsize=gridsize, d_grid = .2, 
                        locs_markersize=10, CP_markersize=8, grid_markersize=3, 
                        grid_opacity=1, lines_per_CP=1)
else:
    print('I: No Spline Mapping used')

print('Done')