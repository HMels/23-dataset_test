# catmull_rom_splines

import tensorflow as tf
import matplotlib.pyplot as plt

from MinEntropy_direct import Rel_entropy
import generate_data
from setup_image import Deform

#%% Import dataset
path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]

locs_A, locs_B = generate_data.run_channel_generation(
    path, deform=Deform(), error=0, Noise=0, realdata=True, subset=1, pix_size=1
    )

ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)


#%% Create ControlPoints
gridsize = 50               # nm, size between controlpoints

x1_grid = tf.range(tf.reduce_min(ch2[:,0])-gridsize, tf.reduce_max(ch2[:,0])+2*gridsize, gridsize)
x2_grid = tf.range(tf.reduce_min(ch2[:,1])-gridsize, tf.reduce_max(ch2[:,1])+2*gridsize, gridsize)
CP_locs = tf.stack(tf.meshgrid(x1_grid, x2_grid  ), axis=2) # control points locations
CP_idx = tf.cast(tf.stack([( ch2[:,1]-tf.reduce_min(ch2[:,1]) )//gridsize+1, 
                           ( ch2[:,0]-tf.reduce_min(ch2[:,0]) )//gridsize+1], axis=1),
                 dtype=tf.int32) # the index of which CP the points in ch2 belong to



plt.close('all')
CP_corners = tf.gather_nd(CP_locs,CP_idx)
plt.plot(CP_corners[:,0],CP_corners[:,1], 'x', label='Control Points')
plt.plot(ch2[:,0], ch2[:,1], 'o', label='Localizations')
plt.legend()


#%% Sum A
def Sum_A(a,b, CP_locs, CP_idx):
    A = tf.Variable([
    [-.5, 1.5, -1.5, 0.5],
    [1, -2.5, 2, -.5],
    [-.5, 0, -.5, 0],
    [-.5, 0, .5, 0],
    [0, 1, 0, 0]
    ], trainable=False, dtype=tf.float32) 
    
    q00 = tf.gather_nd(CP_locs, CP_idx+[-1,-1])  # q_k
    q01 = tf.gather_nd(CP_locs, CP_idx+[-1,0])  # q_k
    q02 = tf.gather_nd(CP_locs, CP_idx+[-1,1])  # q_k
    q03 = tf.gather_nd(CP_locs, CP_idx+[-1,2])  # q_k
    
    q10 = tf.gather_nd(CP_locs, CP_idx+[0,-1])  # q_k
    q11 = tf.gather_nd(CP_locs, CP_idx+[0,0])  # q_k
    q12 = tf.gather_nd(CP_locs, CP_idx+[0,1])  # q_k
    q13 = tf.gather_nd(CP_locs, CP_idx+[0,2])  # q_k
    
    q20 = tf.gather_nd(CP_locs, CP_idx+[1,-1])  # q_k
    q21 = tf.gather_nd(CP_locs, CP_idx+[1,0])  # q_k
    q22 = tf.gather_nd(CP_locs, CP_idx+[1,1])  # q_k
    q23 = tf.gather_nd(CP_locs, CP_idx+[1,2])  # q_k
    
    q30 = tf.gather_nd(CP_locs, CP_idx+[2,-1])  # q_k
    q31 = tf.gather_nd(CP_locs, CP_idx+[2,0])  # q_k
    q32 = tf.gather_nd(CP_locs, CP_idx+[2,1])  # q_k
    q33 = tf.gather_nd(CP_locs, CP_idx+[2,2])  # q_k
    
    
    A_matrix = tf.stack([
        A[a,0]*A[b,0]*q00,
        A[a,0]*A[b,1]*q01,
        A[a,0]*A[b,2]*q02,
        A[a,0]*A[b,3]*q03,
        
        A[a,1]*A[b,0]*q10,
        A[a,1]*A[b,1]*q11,
        A[a,1]*A[b,2]*q12,
        A[a,1]*A[b,3]*q13,
        
        A[a,2]*A[b,0]*q20,
        A[a,2]*A[b,1]*q21,
        A[a,2]*A[b,2]*q22,
        A[a,2]*A[b,3]*q23,
    
        A[a,3]*A[b,0]*q30,
        A[a,3]*A[b,1]*q31,
        A[a,3]*A[b,2]*q32,
        A[a,3]*A[b,3]*q33
        ], axis=2) 
    return tf.reduce_sum(A_matrix, axis=2)


#%% CatmullRomSplines
x_input = tf.Variable([[.5, 1.5], [1.5, 0.5]], dtype=tf.float32)
x1 = tf.Variable([-1,0,1,2,3], dtype=tf.float32)
CP_locs = tf.stack(tf.meshgrid(x1,x1), axis=2)
CP_idx = tf.Variable([[2,1],[1,2]])    
    
q11 = tf.gather_nd(CP_locs, CP_idx)
r = x_input - q11
x = r[:,0][:,None]
y = r[:,1][:,None]

M_matrix = tf.stack([
   tf.pow(x,0)*tf.pow(y,0)*Sum_A(0,0, CP_locs, CP_idx),
   tf.pow(x,0)*tf.pow(y,1)*Sum_A(0,1, CP_locs, CP_idx),
   tf.pow(x,0)*tf.pow(y,2)*Sum_A(0,2, CP_locs, CP_idx),
   tf.pow(x,0)*tf.pow(y,3)*Sum_A(0,3, CP_locs, CP_idx),
   
   tf.pow(x,1)*tf.pow(y,0)*Sum_A(1,0, CP_locs, CP_idx),
   tf.pow(x,1)*tf.pow(y,1)*Sum_A(1,1, CP_locs, CP_idx),
   tf.pow(x,1)*tf.pow(y,2)*Sum_A(1,2, CP_locs, CP_idx),
   tf.pow(x,1)*tf.pow(y,3)*Sum_A(1,3, CP_locs, CP_idx),
        
   tf.pow(x,2)*tf.pow(y,0)*Sum_A(2,0, CP_locs, CP_idx),
   tf.pow(x,2)*tf.pow(y,1)*Sum_A(2,1, CP_locs, CP_idx),
   tf.pow(x,2)*tf.pow(y,2)*Sum_A(2,2, CP_locs, CP_idx),
   tf.pow(x,2)*tf.pow(y,3)*Sum_A(2,3, CP_locs, CP_idx),
   
   tf.pow(x,3)*tf.pow(y,0)*Sum_A(3,0, CP_locs, CP_idx),
   tf.pow(x,3)*tf.pow(y,1)*Sum_A(3,1, CP_locs, CP_idx),
   tf.pow(x,3)*tf.pow(y,2)*Sum_A(3,2, CP_locs, CP_idx),
   tf.pow(x,3)*tf.pow(y,3)*Sum_A(3,3, CP_locs, CP_idx),
   ], axis=2) 
x_mapped = tf.reduce_sum(M_matrix, axis=2)
        
plt.figure()
plt.plot(x_mapped-x_input, 'x')
print(x_mapped)
print(x_input)
