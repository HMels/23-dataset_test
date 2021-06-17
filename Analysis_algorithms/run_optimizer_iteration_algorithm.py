# run_optimizer_iteration_algorithm.py
'''
This program is written to investigate how different optimizers converge 
in a model. It plots the average error vs the iterations. This program utilises 
the HEL1 and Beads Mimics
'''

# Packages
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import tensorflow as tf

# Classes
from LoadDataModules.Deform import Deform

# Modules
import LoadDataModules.generate_data as generate_data
import MinEntropyModules.Module_ShiftRot as Module_ShiftRot
import MinEntropyModules.train_model as train_model

learning_rates=[1]
opts=[
      tf.optimizers.Adam,
      tf.optimizers.Adagrad,
      tf.optimizers.Adadelta,
      tf.optimizers.Adamax,
      tf.optimizers.Ftrl,
      tf.optimizers.Nadam
      ]

opts_name=[
      'Adam',
      'Adagrad',
      'Adadelta',
      'Adamax',
      'Ftrl',
      'Nadam'
      ]


#%%
Ni=len(learning_rates)
Nj=len(opts)

N_points=60
x = np.unique(np.logspace(0,np.log(3000)/np.log(10),N_points+1).astype('int'))
N = np.max(x)

N_it = 10
dist_avg = np.empty([N_it, Ni,Nj,x.shape[0]+1])
dist_avg[:,:]=np.nan

#%%
tf.config.run_functions_eagerly(True) 
for l in range(N_it): # Generate N_it different datasets to calculate the averages of

    shift = np.array([ 20  , 20 ]) + 10*rnd.randn(2)                     # shift in nm
    rotation = 0.2*rnd.randn(1)                 # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
    deform = Deform(shift, rotation)
        
    #locs_A, locs_B = generate_data.generate_beads_mimic(deform, Nlocs[i], error=error, Noise=Noise)
    locs_A, locs_B = generate_data.generate_HEL1_mimic(Nclust=100, deform=deform,
                                                           error=.0, Noise=.0)
        
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)

    model=Module_ShiftRot.ShiftRotMod_direct()
    
    for i in range(Ni): # iterate over the learning rates
        for j in range(Nj): # iterate over the optimizers
        
            # original distance
            ch2_copy=tf.Variable(ch2)
            dist = np.sqrt((ch2_copy-ch1)[:,0]**2 + (ch2_copy-ch1)[:,1]**2)
            dist_avg[l,i,j,0] = np.average(dist)
            del ch2_copy
            
            mods = train_model.Models(model=model, learning_rate=learning_rates[i], 
                              opt=opts[j], threshold=10)
            
            m=0
            for k in range(N): # Train
                
                mods.Training_loop(ch1, ch2)
                    
                # Calculate the error
                if k==x[m]-1:
                    ch2_copy=tf.Variable(ch2)
                    ch2_map = mods.model.transform_vec(ch2_copy)
                    dist = np.sqrt((ch2_map-ch1)[:,0]**2 + (ch2_map-ch1)[:,1]**2)
                    dist_avg[l,i,j,m+1] = np.average(dist)
                    m+=1
                    del ch2_copy
                    
                if (k+1)%100==0:
                    print('(N_it, lr, opt, train)=(',l+1,'/',N_it,'-',i+1,'/',Ni,'-',j+1,'/',Nj,'-',k+1,'/',N,')')
                    
            del mods
    del model, ch1, ch2, locs_A, locs_B
        
        
#%%
plt.close('all')
dist_avg_avg=np.average(dist_avg, axis=0)
dist_avg_std=np.std(dist_avg, axis=0)

color = ['blue','red','green','navy','orange','purple','pink','brown']
x_new = np.concatenate([[.9], x],axis=0)

ymin=np.min(dist_avg_avg)/2
ymax=np.max(dist_avg_avg)*2
for i in range(Ni):
    fig, ax = plt.subplots(nrows = 3, ncols = 2)
    title='Comparisson between optimizers for learning-rate='+str(learning_rates[i])
    fig.suptitle(title)
    for j in range(Nj):
        j_mod = j%3
        j_dev = j//3
        ax[j_mod,j_dev].errorbar(x_new, dist_avg_avg[i,j,:], yerr=dist_avg_std[i,j,:], label=str(opts_name[j]),
                     ls=':',fmt='', color=color[j], ecolor=color[j], capsize=3)
        
       # ax[j_mod,j_dev].set_title(str(opts[j]))
        #ax[j_mod,j_dev].legend()
        ax[j_mod,j_dev].set_xlim(x_new[0],x_new[-1])
        ax[j_mod,j_dev].set_ylim(ymin,ymax)
        ax[j_mod,j_dev].set_xscale('log')
        ax[j_mod,j_dev].set_yscale('log')
        ax[j_mod,j_dev].legend()
        
        if j_mod==2: ax[j_mod,j_dev].set_xlabel('Iterations')
        if j_dev==0: ax[j_mod,j_dev].set_ylabel('Average Error')
    #fig.legend()