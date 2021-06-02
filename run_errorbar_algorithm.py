# run_errorbar_algorithm
# Packages
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Classes
from setup_image import Deform

# Modules
import generate_data
import run_optimization
import output_fn
import generate_image

#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)
    

#%% Error shift
## Dataset
realdata = False                                    # load real data or generate from real data
direct = True                                       # True if data is coupled
subset = .1                                          # percentage of original dataset
pix_size = 100
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]
#path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]

## System Parameters
error = 0.0                                         # localization error in nm
Noise = 0.0                                         # percentage of noise

N=40
shift_max = 100
shift = np.array([ np.linspace(0,shift_max,N), np.zeros(N)])
calc_shift = np.zeros(N)
rotation = .5                                      # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)

dist = []
for i in range(N):
    deform = Deform(shift[:,i], rotation)

    locs_A, locs_B = generate_data.run_channel_generation(
                    path, deform, error, Noise, realdata, subset, pix_size
                    )
    
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)
    ch2_map = tf.Variable(ch2)
    
    # training loop ShiftRotMod
    mods1, ch2_map = run_optimization.run_optimization_ShiftRot(ch1, ch2_map, maxDistance=30, 
                                                                threshold=10, learning_rate=1,
                                                                direct=direct) 
    
    dist_sq = np.sqrt((ch2_map-ch1)[:,0]**2 + (ch2_map-ch1)[:,1]**2)
    dist.append(np.average(dist_sq))
    calc_shift[i]=mods1.model.trainable_variables[0].numpy()[0]
    
    del mods1, ch2, ch2_map, locs_A, locs_B, deform

#%% 
##plotting
plt.close('all')
plt.figure()
plt.plot(shift[0,:], dist, 'r+')
plt.title('Error vs Shift (direct, 0.5 degrees rotation)')
plt.xlabel('Shift [nm]')
plt.ylabel('Error [nm]')
plt.xlim([0,shift_max])
plt.ylim(0)

#%% Error N
## Dataset
realdata = False                                    # load real data or generate from real data
direct = True                                       # True if data is coupled
pix_size = 100
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 
          'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]

## System Parameters
error = 0.0                                         # localization error in nm
Noise = 0.0                                         # percentage of noise

N=20
subset = np.linspace(.1,1,N)
N0 = np.zeros(N)

shift = np.array([ 12  , 9 ])                      # shift in nm
rotation = .05                                      # angle of rotation in degrees (note that we do it times 100 so that the learning rate is correct relative to the shift)
shear = np.array([0,0])                      # shear
scaling = np.array([1,1 ])                    # scaling 
deform = Deform(shift, rotation, shear, scaling)

dist = []
for i in range(N):
    locs_A, locs_B = generate_data.run_channel_generation(
                path, deform, error, Noise, realdata, subset[i], pix_size
                )
    
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)
    
    ch2_map = tf.Variable(ch2)
    
    # training loop ShiftRotMod
    mods1, ch2_map = run_optimization.run_optimization_ShiftRot(ch1, ch2_map, maxDistance=30, 
                                                                threshold=10, learning_rate=1,
                                                                direct=direct) 
    
    dist_sq = np.sqrt((ch2_map-ch1)[:,0]**2 + (ch2_map-ch1)[:,1]**2)
    dist.append(np.average(dist_sq))
    N0[i] = ch2_map.shape[0]
    
    
#%% 
##plotting
plt.figure()
plt.plot(N0, dist, 'r+')
plt.title('Error vs Number of points (direct, .5 degrees rotation)')
plt.xlabel('N')
plt.ylabel('Error [nm]')
plt.xlim(0)
plt.ylim(0)