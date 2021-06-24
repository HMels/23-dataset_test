# run_errorbar_algorithm.py
'''
This program is written to investigate how the average error is dependend on the 
initial deformations (shift) present in the dataset. It utilises both the HEL1 and 
Beads mimics 
'''
import sys
sys.path.append('../')

# Packages
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy.random as rnd
import time

# Classes
from LoadDataModules.Deform import Deform

# Modules
import Model
import LoadDataModules.generate_data as generate_data
import MinEntropyModules.Module_ShiftRot as Module_ShiftRot

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)


#%% Error shift
## System Parameters
error = 0.0                                         # localization error in nm
Noise = 0.0                                         # percentage of noise
N_it = 75

#%%
dist_avg = np.zeros([2,N_it]) # the first row contains the distance before, the second row the distance after
std_avg = np.zeros([2,N_it])
t_lapsed = np.zeros([N_it])
for i in range(N_it):
    ## generate data    
    deform = Deform(
        deform_on=True,                         # True if we want to give channels deform by hand
        shift = np.array([ 20  , 20 ]) + 10*rnd.randn(2),                     # shift in nm        
        rotation = 0.2*rnd.randn(1),                 # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
        shear=np.array([0.003, 0.002])  + 0.001*rnd.randn(2),         # shear
        scaling=np.array([1.0004,1.0003 ])+ 0.0001*rnd.randn(2)    # scaling
        )
    
    #locs_A, locs_B = generate_data.generate_beads_mimic(deform, 216, error=error, Noise=Noise)
    locs_A, locs_B = generate_data.generate_HEL1_mimic(Nclust=650, deform=deform,
                                                       error=error, Noise=Noise)
    
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)
    
    
    ## Metric Pre 
    print('i=',i+1,'/',N_it)
    dist = np.sqrt((ch2-ch1)[:,0]**2 + (ch2-ch1)[:,1]**2)
    dist_avg[0,i] = np.average(dist)
    std_avg[0,i] = np.std(dist)
    start = time.time()
    
    
    mods, ch2_map = Model.run_model(ch1, ch2, coupled=True, N_it=[400, 200], 
                                    learning_rate=[1,1e-2], gridsize = 1000)
    
    # Metrics Post
    t_lapsed[i]=time.time()-start
    dist = np.sqrt((ch2_map-ch1)[:,0]**2 + (ch2_map-ch1)[:,1]**2)
    dist_avg[1,i] = np.average(dist)
    std_avg[1,i] = np.std(dist)
    del mods, deform, ch1, ch2, ch2_map
        
        
#%% 
# Plotting the histogram
fig, ax = plt.subplots(nrows = 2, ncols = 2)
#fig.suptitle('The Average and Standard Deviation of the Error')

nbins=60
for i in range(2):
    
    ## The Average Error
    n = ax[0,i].hist(dist_avg[i,:], alpha=.8, edgecolor='red', bins=nbins)
    ymax = np.max(n[0]) +50
    avg1=np.average(dist_avg[i,:])
    ax[0,i].vlines(avg1, color='green', ymin=0, ymax=ymax, label=('avg='+str(round(avg1,2))))
                
    # Some extra plotting parameters
    ax[0,i].set_ylim([0,ymax])
    ax[0,i].set_xlim(0)
    ax[0,i].set_xlabel('Average Error [nm]')
    ax[0,i].set_ylabel('# of Simulations')
    ax[0,i].legend()
    
    ## The Standard Deviation of the Error
    n = ax[1,i].hist(std_avg[i,:], alpha=.8, edgecolor='blue', color='orange', bins=nbins)
    ymax = np.max(n[0]) + 50
    avg1=np.average(std_avg[i,:])
    ax[1,i].vlines(avg1, color='green', ymin=0, ymax=ymax, label=('avg='+str(round(avg1,2))))
                
    # Some extra plotting parameters
    ax[1,i].set_ylim([0,ymax])
    ax[1,i].set_xlim(0)
    ax[1,i].set_xlabel('Std of the Error [nm]')
    ax[1,i].set_ylabel('# of Simulations')
    ax[1,i].legend()

ax[0,0].set_title('Original')
ax[0,1].set_title('Mapped')
fig.show()

