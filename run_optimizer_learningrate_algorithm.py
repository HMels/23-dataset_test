# run_optimizer_learningrate_algorithm
# Packages
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Classes
from setup_image import Deform

# Modules
import generate_data
import Module_ShiftRot
import train_model

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)

#%% Error N
## System Parameters
error = 0.0                                         # localization error in nm
Noise = 0.0                                         # percentage of noise

#path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
#          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]

locs_A, locs_B = generate_data.generate_channels(
    path, deform=Deform(), error=error, Noise=Noise, realdata=True, subset=1, pix_size=1
    )
ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)

model=Module_ShiftRot.ShiftRotMod_direct()

learning_rates=[1]
opts=[
      tf.optimizers.Adam,
      tf.optimizers.Adagrad,
      tf.optimizers.Adadelta,
      tf.optimizers.Adamax,
      tf.optimizers.Ftrl,
      tf.optimizers.Nadam,
      #tf.optimizers.SGD,
      tf.optimizers.RMSprop
      ]

#%%
Ni=len(learning_rates)
Nj=len(opts)
N=40
dx=1
dist_avg = np.empty([Ni,Nj,int(N/dx)+1])
dist_avg[:,:]=np.nan

tf.config.run_functions_eagerly(True) 
for i in range(Ni):
    for j in range(Nj):
        # original distance
        ch2_copy=tf.Variable(ch2)
        dist = np.sqrt((ch2_copy-ch1)[:,0]**2 + (ch2_copy-ch1)[:,1]**2)
        dist_avg[i,j,0] = np.average(dist)
        del ch2_copy
        
        mods = train_model.Models(model=model, learning_rate=learning_rates[i], 
                          opt=opts[j], threshold=10)
        for k in range(N):
            mods.Training_loop(ch1, ch2)
                
            # Calculate the error
            if k%dx==0:
                ch2_copy=tf.Variable(ch2)
                ch2_map = mods.model.transform_vec(ch2_copy)
                dist = np.sqrt((ch2_map-ch1)[:,0]**2 + (ch2_map-ch1)[:,1]**2)
                dist_avg[i,j,int(k/dx)+1] = np.average(dist)
                del ch2_copy
                
            # break loop after a certain amount of rejections
            if mods.endloop:
                break
                    
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
    for j in range(Nj):
        plt.plot(x, dist_avg[i,j,:],label=str(opts[j]))
    plt.legend()
    plt.xlim([0,N])
    plt.ylim(0)
