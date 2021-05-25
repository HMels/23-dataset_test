# catmull_rom_splines

import tensorflow as tf
import matplotlib.pyplot as plt

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

x1_grid = tf.range(tf.reduce_min(ch2[:,0])-1.5*gridsize,
                   tf.reduce_max(ch2[:,0])+2*gridsize, gridsize)
x2_grid = tf.range(tf.reduce_min(ch2[:,1])-1.5*gridsize, 
                   tf.reduce_max(ch2[:,1])+2*gridsize, gridsize)
CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
CP_idx = tf.cast(tf.stack([( ch2[:,0]-tf.reduce_min(ch2[:,0]) )//gridsize+1, 
                           ( ch2[:,1]-tf.reduce_min(ch2[:,1]) )//gridsize+1], axis=1),
                 dtype=tf.int32) 



plt.close('all')
CP_corners = tf.gather_nd(CP_locs,CP_idx)
plt.plot(CP_corners[:,0],CP_corners[:,1], 'x', label='Control Points')
plt.plot(ch2[:,0], ch2[:,1], 'o', label='Localizations')
plt.legend()


#%% Sum A
def Sum_A(a,b, W):
    A = tf.Variable([
    [-.5, 1.5, -1.5, 0.5],
    [1, -2.5, 2, -.5],
    [-.5, 0, .5, 0],
    [0, 1, 0, 0]
    ], trainable=False, dtype=tf.float32) 
    
    A_matrix = tf.stack([
        A[a,0]*A[b,0]*W[0],
        A[a,0]*A[b,1]*W[1],
        A[a,0]*A[b,2]*W[2],
        A[a,0]*A[b,3]*W[3],
        
        A[a,1]*A[b,0]*W[4],
        A[a,1]*A[b,1]*W[5],
        A[a,1]*A[b,2]*W[6],
        A[a,1]*A[b,3]*W[7],
        
        A[a,2]*A[b,0]*W[8],
        A[a,2]*A[b,1]*W[9],
        A[a,2]*A[b,2]*W[10],
        A[a,2]*A[b,3]*W[11],
    
        A[a,3]*A[b,0]*W[12],
        A[a,3]*A[b,1]*W[13],
        A[a,3]*A[b,2]*W[14],
        A[a,3]*A[b,3]*W[15]
        ], axis=2) 
    return tf.reduce_sum(A_matrix, axis=2)

   
def Splines(CP_locs, CP_idx):
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
    
    return [q00, q01, q02, q03, 
            q10, q11, q12, q13,
            q20, q21, q22, q23, 
            q30, q31, q32, q33]


#%% CatmullRomSplines
'''
gridsize = 1
x_input_original = tf.Variable([[.5, 1.5], [1.5, 0.5], [1.5, 1.5], [0.5, 0.5],
                       [3.5,4.5], [0.5,4.5], [0.5,3.5], [3.5,1.5]], dtype=tf.float32)
'''
x_input = ch2 / gridsize
x1_grid = tf.range(tf.reduce_min(tf.floor(x_input[:,0])) -1,
                   tf.reduce_max(tf.floor(x_input[:,0])) +3, 1)
x2_grid =  tf.range(tf.reduce_min(tf.floor(x_input[:,1]))-1,
                   tf.reduce_max(tf.floor(x_input[:,1])) +3, 1)
CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
CP_idx = tf.cast(tf.stack(
    [( x_input[:,0]-tf.reduce_min(tf.floor(x_input[:,0]))+1)//1 , 
     ( x_input[:,1]-tf.reduce_min(tf.floor(x_input[:,1]))+1)//1 ], 
    axis=1), dtype=tf.int32)

W = Splines(CP_locs, CP_idx)

r = x_input-W[5]
x = r[:,0][:,None]
y = r[:,1][:,None]

M_matrix = tf.stack([
   tf.pow(x,3)*tf.pow(y,3)*Sum_A(0,0, W),
   tf.pow(x,3)*tf.pow(y,2)*Sum_A(0,1, W),
   tf.pow(x,3)*tf.pow(y,1)*Sum_A(0,2, W),
   tf.pow(x,3)*tf.pow(y,0)*Sum_A(0,3, W),
   
   tf.pow(x,2)*tf.pow(y,3)*Sum_A(1,0, W),
   tf.pow(x,2)*tf.pow(y,2)*Sum_A(1,1, W),
   tf.pow(x,2)*tf.pow(y,1)*Sum_A(1,2, W),
   tf.pow(x,2)*tf.pow(y,0)*Sum_A(1,3, W),
        
   tf.pow(x,1)*tf.pow(y,3)*Sum_A(2,0, W),
   tf.pow(x,1)*tf.pow(y,2)*Sum_A(2,1, W),
   tf.pow(x,1)*tf.pow(y,1)*Sum_A(2,2, W),
   tf.pow(x,1)*tf.pow(y,0)*Sum_A(2,3, W),
   
   tf.pow(x,0)*tf.pow(y,3)*Sum_A(3,0, W),
   tf.pow(x,0)*tf.pow(y,2)*Sum_A(3,1, W),
   tf.pow(x,0)*tf.pow(y,1)*Sum_A(3,2, W),
   tf.pow(x,0)*tf.pow(y,0)*Sum_A(3,3, W),
   ], axis=2) 
x_mapped = tf.reduce_sum(M_matrix, axis=2)

        
plt.close('all')
plt.figure()
plt.plot(x_mapped[:,1]*gridsize,x_mapped[:,0]*gridsize, 'r.', label='Mapped')
plt.plot(x_input[:,1]*gridsize,x_input[:,0]*gridsize, 'b.', label='Original')
plt.plot(W[5][:,1]*gridsize,W[5][:,0]*gridsize,'rx',label='Floor')
plt.plot(CP_locs[0,:,1]*gridsize, CP_locs[0,:,0]*gridsize, 'y+', label='Grid')
for i in range(1,CP_locs.shape[0]):
    plt.plot(CP_locs[i,:,1]*gridsize, CP_locs[i,:,0]*gridsize, 'y+')
plt.legend()
print(x_mapped*gridsize)
print(x_input*gridsize)

plt.figure()
plt.plot(x_mapped[:,1]-x_input[:,1], x_mapped[:,0]-x_input[:,0], 'x')