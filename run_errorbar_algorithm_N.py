# run_errorbar_algorithm
# Packages
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy.random as rnd
import time

# Classes
from setup_image import Deform

# Modules
import generate_data
import run_optimization

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)

#%% Error N
## System Parameters
error = 0.0                                         # localization error in nm
Noise = 0.0                                         # percentage of noise

N=20
N_it = 10
Nlocs = np.logspace(1,5,N).astype('int')

#%%
dist_avg_N = np.zeros([N,N_it])
t_lapsed = np.zeros([N,N_it])
for i in range(N):
    for j in range(N_it):
        print('i=',i,'/',N,' j=',j,'/',N_it)
        start = time.time()
        
        shift = np.array([ 20  , 20 ]) + 10*rnd.randn(2)                     # shift in nm
        rotation = 0.2*rnd.randn(1)                 # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
        deform = Deform(shift, rotation)
        
        locs_A, locs_B = generate_data.generate_channels_random(Nlocs[i], deform, error=error, Noise=Noise)
        
        ch1 = tf.Variable( locs_A, dtype = tf.float32)
        ch2 = tf.Variable( locs_B, dtype = tf.float32)
        ch2_map = tf.Variable(ch2)
        
        # training loop ShiftRotMod
        mods1, ch2_map = run_optimization.run_optimization_ShiftRot(ch1, ch2_map, maxDistance=30, 
                                                                    threshold=10, learning_rate=1,
                                                                    direct=True) 
        
        dist = np.sqrt((ch2_map-ch1)[:,0]**2 + (ch2_map-ch1)[:,1]**2)
        dist_avg_N[i,j] = np.average(dist)
    
        t_lapsed[i,j]=time.time()-start
        del mods1, deform
    
#%% 
avg = np.average(dist_avg_N, axis=1)
std = np.std(dist_avg_N, axis=1)
t = np.average(t_lapsed, axis=1)

#%% Plotting
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.set_title('Average Error vs Number of points (direct, norm(0.2) degrees rotation, 20+norm(10) nm shift for N_it='+str(N_it)+') for Shift and Rotation')
ax.set_xlabel('Number of locs per Channel')
ax.set_ylabel('Error [nm]')
ax2.set_ylabel('time [s]')

p1 = ax.errorbar(Nlocs, avg, yerr=std, ls=':', label='Average Error')
p2 = ax2.bar(Nlocs, t, label='Average Time', width=0.3*Nlocs, align='center', alpha=0.55, 
        edgecolor='red', color='orange')
ax.set_xscale('log')
ax.legend(handles=[p1, p2], loc='best')
ax.set_xlim(np.min(Nlocs)-3)
ax.set_ylim(0)
ax2.set_ylim(0,3*np.max(t))