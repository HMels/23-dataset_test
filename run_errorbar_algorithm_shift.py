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
    

#%% Error shift
## System Parameters
error = 0.0                                         # localization error in nm
Noise = 0.0                                         # percentage of noise

N=40
N_it = 10
Nlocs1 = 2000
shift_max = 40
shift = np.array([ np.linspace(0,shift_max,N), np.zeros(N)])

#%%
calc_shift = np.zeros([N,N_it])
dist_avg = np.zeros([N,N_it])
t_lapsed = np.zeros([N,N_it])
for i in range(N):
    for j in range(N_it):
        print('i=',i,'/',N,' j=',j,'/',N_it)
        start = time.time()
        
        deform = Deform(shift[:,i], 0.2*rnd.randn(1))
        locs_A, locs_B = generate_data.generate_channels_random(Nlocs1, deform, error=error, Noise=Noise)
    
        
        ch1 = tf.Variable( locs_A, dtype = tf.float32)
        ch2 = tf.Variable( locs_B, dtype = tf.float32)
        
        # training loop ShiftRotMod
        mods1, ch2_map = run_optimization.run_optimization_ShiftRot(ch1, ch2, maxDistance=30, 
                                                                    threshold=10, learning_rate=1, direct=True) 
        
        # metrics post
        dist = np.sqrt((ch2_map-ch1)[:,0]**2 + (ch2_map-ch1)[:,1]**2)
        dist_avg[i,j] = np.average(dist)
        calc_shift[i,j]=mods1.model.trainable_variables[0].numpy()[0]
        
        t_lapsed[i,j]=time.time()-start
        del mods1, deform
        
        
#%% 
error=np.zeros(calc_shift.shape)
for i in range(calc_shift.shape[0]):
    error[i,:]=np.abs(calc_shift[i,:]+shift[0,i])
avg_shift = np.average(error, axis=1)
std_shift = np.std(error, axis=1)
avg2 = np.average(dist_avg, axis=1)
std2 = np.std(dist_avg, axis=1)
t2 = np.average(t_lapsed, axis=1)

#%%plotting
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.set_title('Error vs Shift (direct, norm(0.2) rotation, N='+str(Nlocs1)+' and with N_it='+str(N_it)+') for Shift and Rotation')
ax.set_xlabel('Shift [nm]')
ax.set_ylabel('Error [nm]')
ax2.set_ylabel('time [s]')

p1=ax.errorbar(shift[0,:], avg_shift, yerr=std_shift, ls=':',label='Absolute Error of calculated Shift', color='blue')
p2=ax.errorbar(shift[0,:]+.1, avg2, yerr=std2, ls=':', label='Average Error', color='green')
w = (shift[0,1]-shift[0,0])/4
p3=ax2.bar(shift[0,:]+.2, t2, label='Average Time', width=w, align='edge', alpha=0.55, 
           edgecolor='red', color='orange')
ax.legend(handles=[p1, p2, p3], loc='best')
ax.set_xlim([0,shift_max])
ax.hlines(0,0,shift_max, color='black', linewidth=0.5)
ax.set_ylim(0)
ax2.set_ylim(0,1.5*np.max(t2))