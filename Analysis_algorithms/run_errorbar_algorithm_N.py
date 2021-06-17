# run_errorbar_algorithm.py
'''
This program is written to investigate how the average error is dependend on the 
number of localizations present in the dataset. It utilises both the HEL1 and 
Beads mimics 
'''


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
import LoadDataModules.generate_data as generate_data
import MinEntropyModules.Module_ShiftRot as Module_ShiftRot

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
Nlocs = np.logspace(0,3,N).astype('int')

#%%
dist_avg_N = np.zeros([N,N_it])
t_lapsed = np.zeros([N,N_it])
N_tot = np.zeros([N,N_it])
for i in range(N):
    for j in range(N_it):
        print('i=',i+1,'/',N,' j=',j+1,'/',N_it)
        start = time.time()
        
        shift = np.array([ 20  , 20 ]) + 10*rnd.randn(2)                     # shift in nm
        rotation = 0.2*rnd.randn(1)                 # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
        deform = Deform(shift, rotation)
        
        #locs_A, locs_B = generate_data.generate_beads_mimic(deform, Nlocs[i], error=error, Noise=Noise)
        locs_A, locs_B = generate_data.generate_HEL1_mimic(Nclust=Nlocs[i], deform=deform,
                                                           error=error, Noise=Noise)
        
        ch1 = tf.Variable( locs_A, dtype = tf.float32)
        ch2 = tf.Variable( locs_B, dtype = tf.float32)
        ch2_map = tf.Variable(ch2)
        
        # training loop ShiftRotMod
        mods1, ch2_map = Module_ShiftRot.run_optimization(ch1, ch2_map, maxDistance=30, 
                                                          threshold=10, learning_rate=1, direct=True) 
        
        dist = np.sqrt((ch2_map-ch1)[:,0]**2 + (ch2_map-ch1)[:,1]**2)
        dist_avg_N[i,j] = np.average(dist)
    
        t_lapsed[i,j]=time.time()-start
        N_tot[i,j]=ch1.shape[0]
        del mods1, deform
    
#%% 
avg = np.average(dist_avg_N, axis=1)
std = np.std(dist_avg_N, axis=1)
t = np.average(t_lapsed, axis=1)
N_avg = np.average(N_tot, axis=1)
N_std = np.std(N_tot, axis=1)

#%% Plotting
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.set_title('Average Error vs Number of points (direct, norm(0.2) degrees rotation, 20+norm(10) nm shift for N_it='+str(N_it)+') for Shift and Rotation')
ax.set_xlabel('Number of locs per Channel')
ax.set_ylabel('Error [nm]')
ax2.set_ylabel('time [s]')

p1 = ax.errorbar(N_avg, avg, yerr=std, xerr=N_std, ls=':', fmt='', ecolor='black', capsize=5, label='Average Error')
p2 = ax2.bar(N_avg*1.05, t, label='Average Time', width=20*Nlocs, align='center', alpha=0.55, 
        edgecolor='red', color='orange')
ax.set_xscale('log')
ax.legend(handles=[p1, p2], loc='best')
ax.set_xlim(np.min(Nlocs)-3)
ax.set_ylim(0)
ax2.set_ylim(0,3*np.max(t))