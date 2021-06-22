# run_optimizer_learningrate_algorithm
'''
This program investigates how different optimizers converge according to different
learning rates in a model. It is mostly used to get a general idea, and not to be 
used as a quantification as it does not work with averages
'''

# Packages
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Classes
from LoadDataModules.Deform import Deform

# Modules
import LoadDataModules.generate_data as generate_data
import MinEntropyModules.Module_ShiftRot as Module_ShiftRot
import MinEntropyModules.Module_Splines as Module_Splines
import MinEntropyModules.train_model as train_model

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)

#%% run_optimization
def run_optimization(ch1, ch2, N_it=3000, gridsize=50, maxDistance=50, 
                             learning_rate=1e-3, direct=False, opt=tf.optimizers.Adagrad):
    '''
    Parameters
    ----------
    ch1 ,ch2 : Nx2 Tensor
        Tensor containing the localizations of their channels.
    N_it : int, optional
        Number of iterations used in the training loop. The default 
    gridsize : float32, optional
        The size of the grid splines
    maxDistance : float32, optional
        The distance in which the Nearest Neighbours will be searched. 
        The default is 50nm.
    learning_rate : float, optional
        The initial learning rate of our optimizer. the default is 1.
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.
    opt : tf.optimizers, optional
        The Optimizer used. The default is tf.optimizers.Adagrad

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
        
    if direct:          # direct 
        nn1=None
        nn2=None
        
        x1_grid = tf.range(tf.reduce_min(tf.floor(ch2_input[:,0])) -1,
                       tf.reduce_max(tf.floor(ch2_input[:,0])) +3, 1)
        x2_grid =  tf.range(tf.reduce_min(tf.floor(ch2_input[:,1]))-1,
                            tf.reduce_max(tf.floor(ch2_input[:,1])) +3, 1)
        CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
        model = Module_Splines.CatmullRomSplines_direct(CP_locs, CP_idx, ch2)
    
    
    # The Model
    mods = train_model.Models(model=model, learning_rate=learning_rate, opt=opt)
    
    # The Training Function
    model_apply_grads = train_model.get_apply_grad_fn()
    mods, ch2_input = model_apply_grads(ch1_input, ch2_input, N_it, mods, nn1, nn2)
    return mods, ch2_input*gridsize

#%% Error N
## System Parameters
error = 0.0                                         # localization error in nm
Noise = 0.0                                         # percentage of noise
gridsize = 100

path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]
locs_A, locs_B = generate_data.generate_channels(
    path=path, deform=Deform(), error=error, Noise=Noise, copy_channel=False,
    subset=1,                               # the subset of the dataset we want to load
    pix_size=1                              # size of a pixel in nm
    )

ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)

ch1_input = tf.Variable(ch1/gridsize, trainable=False)
ch2_input = tf.Variable(ch2/gridsize, trainable=False)
        
CP_idx = tf.cast(tf.stack(
    [( ch2_input[:,0]-tf.reduce_min(tf.floor(ch2_input[:,0]))+1)//1 , 
     ( ch2_input[:,1]-tf.reduce_min(tf.floor(ch2_input[:,1]))+1)//1 ], 
    axis=1), dtype=tf.int32)

x1_grid = tf.range(tf.reduce_min(tf.floor(ch2_input[:,0])) -1,
                   tf.reduce_max(tf.floor(ch2_input[:,0])) +3, 1)
x2_grid =  tf.range(tf.reduce_min(tf.floor(ch2_input[:,1]))-1,
                    tf.reduce_max(tf.floor(ch2_input[:,1])) +3, 1)
CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])

#model=Module_ShiftRot.ShiftRotMod_direct()
model = Module_Splines.CatmullRomSplines_direct(CP_locs, CP_idx, ch2_input)

learning_rates=[1e-2]
opts=[
      tf.optimizers.Adam,
      tf.optimizers.Adagrad,
      tf.optimizers.Adadelta,
      tf.optimizers.Adamax,
      #tf.optimizers.Ftrl,
      #tf.optimizers.Nadam
      ]

#%%
Ni=len(learning_rates)
Nj=len(opts)
N=500
dx=5
dist_avg = np.empty([Ni,Nj,int(N/dx)+1])
dist_avg[:,:]=np.nan

tf.config.run_functions_eagerly(True) 
for i in range(Ni):
    for j in range(Nj):
        # original distance
        ch2_copy=tf.Variable(ch2_input)
        dist = np.sqrt((ch2_copy-ch1_input)[:,0]**2 + (ch2_copy-ch1_input)[:,1]**2)
        dist_avg[i,j,0] = np.average(dist)
        del ch2_copy
        
        mods = train_model.Models(model=model, learning_rate=learning_rates[i], 
                          opt=opts[j], threshold=10)
        for k in range(N):
            mods.Training_loop(ch1_input, ch2_input)
                
            # Calculate the error
            if k%dx==0:
                ch2_copy=tf.Variable(ch2_input)
                ch2_map = mods.model.transform_vec(ch2_copy)
                dist = np.sqrt((ch2_map-ch1_input)[:,0]**2 + (ch2_map-ch1_input)[:,1]**2)
                dist_avg[i,j,int(k/dx)+1] = np.average(dist)
                del ch2_copy
                
            if (k+1)%100==0:
                print('(i,j,k)=(',i+1,'/',Ni,'-',j+1,'/',Nj,'-',k+1,'/',N,')')
                
        del mods#, ch2_copy
        
        
#%%
plt.close('all')
x=np.arange(0,N+dx,dx)
for i in range(Ni):
    plt.figure()
    title='Comparisson between optimizers for learning-rate='+str(learning_rates[i])
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Average Error')
    for j in range(4):
        plt.plot(x, dist_avg[i,j,:]*gridsize,label=str(opts[j]))
    plt.legend()
    plt.xlim([0,N])
    plt.yscale('log')
    plt.ylim(0)
