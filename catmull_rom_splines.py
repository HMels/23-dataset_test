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

x1_grid = tf.range(tf.reduce_min(ch2[:,0])-gridsize, tf.reduce_max(ch2[:,0])+gridsize, gridsize)
x2_grid = tf.range(tf.reduce_min(ch2[:,1])-gridsize, tf.reduce_max(ch2[:,1])+gridsize, gridsize)
CP_locs = tf.stack(tf.meshgrid(x1_grid, x2_grid  ), axis=2) # control points locations
CP_idx = tf.cast(tf.stack([( ch2[:,1]-tf.reduce_min(ch2[:,1]) )//gridsize+1, 
                           ( ch2[:,0]-tf.reduce_min(ch2[:,0]) )//gridsize], axis=1),
                 dtype=tf.int32) # the index of which CP the points in ch2 belong to


#%% CatmullRomSplines
class CatmullRomSplines(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, CP_locs, CP_idx, name='CatmullRomSplines'):
        super().__init__(name=name)
        
        self.A = tf.Variable([
            [-.5, 1.5, -1.5, 0.5],
            [1, -2.5, 2, -.5],
            [-.5, 0, .5, 0],
            [0, 1, 0, 0]
            ], trainable=False, dtype=tf.float32)
        self.A_mat = self.A[:,None,:]*tf.ones([1,CP_locs.shape[1],1])
        self.CP_locs = tf.Variable(CP_locs, dtype=tf.float32,
                                   trainable=True, name='ControlPoints')
        self.CP_idx = tf.Variable(CP_idx, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
        self.CP_ch2 = tf.stack([
                tf.gather_nd(CP_locs, CP_idx+[-1,0]),
                tf.gather_nd(CP_locs, CP_idx+[0,0]),
                tf.gather_nd(CP_locs, CP_idx+[0,1]),
                tf.gather_nd(CP_locs, CP_idx+[-1,1])
                ]) # matrix containing the corners q_k-1, q_k, q_k+1, q_k+2


    @tf.function
    def call(self, ch1, ch2):
        #ch2_mapped = self.transform_mat( ch2 )
        ch2_mapped = self.transform_vec( ch2 )
        return Rel_entropy(ch1, ch2_mapped)
    
    
    #@tf.autograph.experimental.do_not_convert
    @tf.function
    def transform_vec(self, x_input):
        CP_corners = tf.gather_nd(self.CP_locs, self.CP_idx) # the corners of the grids
        r = x_input - CP_corners
        r_mat = tf.Variable([r**3, r**2, r, r**0], dtype=tf.float32)
        x1= tf.matmul(self.A_mat, self.CP_ch2)
        x_mapped = tf.matmul(r_mat, x1, transpose_a=True)
        return x_mapped
    

    
'''
CP_corners = tf.gather_nd(CP_locs, CP_idx)
A = tf.Variable([[-.5, 1.5, -1.5, 0.5],[1, -2.5, 2, -.5],[-.5, 0, .5, 0],
                    [0, 1, 0, 0]], trainable=False, dtype=tf.float32)
A_mat = A[:,:,None]*tf.ones(ControlPoints.shape[1])
r = ch2 - CP_corners
r_mat = tf.Variable([r**3, r**2, r, r**0], dtype=tf.float32)
x1= tf.matmul(A_mat, ControlPoints)
x_mapped = tf.matmul(r_mat, x1, transpose_a=True)


plt.close('all')
xx= tf.gather_nd(ControlPoints, ControlPointsIdx)
plt.plot(xx-ch2, 'x')


a = tf.constant(np.arange(1, 25, dtype=np.int32),
                shape=[2, 3, 4])
b = tf.constant(np.arange(1, 25, dtype=np.int32),
                shape=[2, 4, 3])
c = tf.matmul(a,b)


'''