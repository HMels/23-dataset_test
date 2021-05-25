# catmull_rom_splines

import tensorflow as tf
import matplotlib.pyplot as plt

import generate_data
from setup_image import Deform
'''
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

x_input = ch2
'''
#%% Sum A
def Sum_A(a,W):
    A = tf.Variable([
    [-.5, 1.5, -1.5, 0.5],
    [1, -2.5, 2, -.5],
    [-.5, 0, .5, 0],
    [0, 1, 0, 0]
    ], trainable=False, dtype=tf.float32) 
    
    A_matrix = tf.stack([
        A[a,0]*W[0],
        A[a,1]*W[1],
        A[a,2]*W[2],
        A[a,3]*W[3]
        ], axis=1) 
    return tf.reduce_sum(A_matrix, axis=1)

   
def Splines(CP_locs, CP_idx):
    q00 = tf.gather(CP_locs, CP_idx-1)  # q_k
    q01 = tf.gather(CP_locs, CP_idx)  # q_k
    q02 = tf.gather(CP_locs, CP_idx+1)  # q_k
    q03 = tf.gather(CP_locs, CP_idx+2)  # q_k
    return [q00, q01, q02, q03]


def Matrix(r, W):
    A = tf.Variable([
    [-.5, 1.5, -1.5, 0.5],
    [1, -2.5, 2, -.5],
    [-.5, 0, .5, 0],
    [0, 1, 0, 0]
    ], trainable=False, dtype=tf.float32) 
    
    ans = []
    for i in range(r.shape[0]):
        Wmat = tf.Variable([[W[0][i], W[1][i], W[2][i], W[3][i]]])
        rmat = tf.Variable([[tf.pow(r[i],3), tf.pow(r[i],2), 
                            tf.pow(r[i],1), tf.pow(r[i],0)]])
        fx = tf.matmul(rmat, tf.matmul(A, Wmat, transpose_b=True))
        ans.append(fx)
    return tf.Variable(ans)


#%% CatmullRomSplines
gridsize = 3
x_input_original = tf.Variable([-0.3, -0.2, -.5, 1, 1.2, 0.5, 3, 3.4, 4],
                               dtype=tf.float32)

x_input = x_input_original / gridsize
CP_locs = tf.range(tf.reduce_min(tf.floor(x_input)) -1,
                   tf.reduce_max(tf.floor(x_input)) +3, 1)
CP_idx = tf.cast( (x_input-tf.reduce_min(tf.floor(x_input))+1)//1, dtype=tf.int32)

W = Splines(CP_locs, CP_idx)

x = x_input-W[1]

M_matrix = tf.stack([
   tf.pow(x,3)*Sum_A(0, W),
   tf.pow(x,2)*Sum_A(1, W),
   tf.pow(x,1)*Sum_A(2, W),
   tf.pow(x,0)*Sum_A(3, W)
   ], axis=1) 
x_mapped = tf.reduce_sum(M_matrix, axis=1)
zero = tf.zeros(len(x_mapped))+0.05
        
x_mapped1 = Matrix(x,W)[:,0,0]

plt.close('all')
plt.figure()
plt.plot(x_mapped*gridsize, zero+.04, 'r.', label='Mapped')
plt.plot(x_mapped1*gridsize, zero+.06, 'g.', label='Mapped1')
plt.plot(x_input*gridsize, zero+.02, 'b.', label='Original')
plt.plot(W[1]*gridsize, zero,'rx',label='Floor')
plt.plot(CP_locs*gridsize, tf.zeros(len(CP_locs))+0.05, 'y+', label='Grid')
plt.legend()
plt.ylim([0,1])
print(x_mapped*gridsize)
print(x_input_original)
