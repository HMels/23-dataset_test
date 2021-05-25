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

x1_grid = tf.range(tf.reduce_min(ch2[:,0])-gridsize, 
                   tf.reduce_max(ch2[:,0])+gridsize, gridsize)
x2_grid = tf.range(tf.reduce_min(ch2[:,1])-gridsize, 
                   tf.reduce_max(ch2[:,1])+gridsize, gridsize)
CP_locs = tf.stack(tf.meshgrid(x1_grid, x2_grid  ), axis=2) # control points locations
CP_idx = tf.cast(tf.stack([( ch2[:,0]-tf.reduce_min(ch2[:,0]) )//gridsize+1, 
                           ( ch2[:,1]-tf.reduce_min(ch2[:,1]) )//gridsize+1], axis=1),
                 dtype=tf.int32) 



plt.close('all')
CP_corners = tf.gather_nd(CP_locs,CP_idx)
plt.plot(CP_corners[:,0],CP_corners[:,1], 'x', label='Control Points')
plt.plot(ch2[:,0], ch2[:,1], 'o', label='Localizations')
plt.legend()
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

        # The location of the ControlPoints. This will be trained
        self.CP_locs = tf.Variable(CP_locs, dtype=tf.float32,
                                   trainable=True, name='ControlPoints')  
        # The indices of which locs in ch2 belong to which CP_locs
        self.CP_idx = tf.Variable(CP_idx, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
        self.A = tf.Variable([
            [-.5, 1.5, -1.5, 0.5],
            [1, -2.5, 2, -.5],
            [-.5, 0, -.5, 0],
            [-.5, 0, .5, 0],
            [0, 1, 0, 0]
            ], trainable=False, dtype=tf.float32)


    @tf.function
    def call(self, ch1, ch2):
        self.reload_q()
        #ch2_mapped = self.transform_mat( ch2 )
        ch2_mapped = self.transform_vec( ch2 )
        return Rel_entropy(ch1, ch2_mapped)
    
    
    #@tf.autograph.experimental.do_not_convert
    @tf.function
    def transform_vec(self, x_input):
        r = x_input - self.q1
        return self.Spline_Map(r[:,0][:,None], r[:,1][:,None])
    
    
    #@tf.autograph.experimental.do_not_convert
    @tf.function
    def Spline_Map(self,x,y):
        M_matrix = tf.stack([
            tf.pow(x,0)*tf.pow(y,0)*self.Sum_A(0,0),
            tf.pow(x,0)*tf.pow(y,1)*self.Sum_A(0,1),
            tf.pow(x,0)*tf.pow(y,2)*self.Sum_A(0,2),
            tf.pow(x,0)*tf.pow(y,3)*self.Sum_A(0,3),
            
            tf.pow(x,1)*tf.pow(y,0)*self.Sum_A(1,0),
            tf.pow(x,1)*tf.pow(y,1)*self.Sum_A(1,1),
            tf.pow(x,1)*tf.pow(y,2)*self.Sum_A(1,2),
            tf.pow(x,1)*tf.pow(y,3)*self.Sum_A(1,3),
            
            tf.pow(x,2)*tf.pow(y,0)*self.Sum_A(2,0),
            tf.pow(x,2)*tf.pow(y,1)*self.Sum_A(2,1),
            tf.pow(x,2)*tf.pow(y,2)*self.Sum_A(2,2),
            tf.pow(x,2)*tf.pow(y,3)*self.Sum_A(2,3),
            
            tf.pow(x,3)*tf.pow(y,0)*self.Sum_A(3,0),
            tf.pow(x,3)*tf.pow(y,1)*self.Sum_A(3,1),
            tf.pow(x,3)*tf.pow(y,2)*self.Sum_A(3,2),
            tf.pow(x,3)*tf.pow(y,3)*self.Sum_A(3,3),
            ], axis=2)
        return tf.reduce_sum(M_matrix, axis=2)
        
    
    
    #@tf.autograph.experimental.do_not_convert
    @tf.function
    def Sum_A(self,a,b):
        A_matrix = tf.stack([
            self.A[a,3]*self.A[b,3]*self.q00,
            self.A[a,3]*self.A[b,2]*self.q01,
            self.A[a,3]*self.A[b,1]*self.q02,
            self.A[a,3]*self.A[b,0]*self.q03,
            
            self.A[a,2]*self.A[b,3]*self.q10,
            self.A[a,2]*self.A[b,2]*self.q11,
            self.A[a,2]*self.A[b,1]*self.q12,
            self.A[a,2]*self.A[b,0]*self.q13,
            
            self.A[a,1]*self.A[b,3]*self.q20,
            self.A[a,1]*self.A[b,2]*self.q21,
            self.A[a,1]*self.A[b,1]*self.q22,
            self.A[a,1]*self.A[b,0]*self.q23,
            
            self.A[a,0]*self.A[b,3]*self.q30,
            self.A[a,0]*self.A[b,2]*self.q31,
            self.A[a,0]*self.A[b,1]*self.q32,
            self.A[a,0]*self.A[b,0]*self.q33
            ], axis=2)
        return tf.reduce_sum(A_matrix, axis=2)
    
    
    @tf.function
    def update_splines(self):
        self.q00 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q01 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q02 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q03 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        
        self.q10 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q11 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q12 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q13 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        
        self.q20 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q21 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q22 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q23 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        
        self.q30 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q31 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q32 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k
        self.q33 = tf.gather_nd(self.CP_locs, self.CP_idx)  # q_k


'''
x_input = ch2
q0 = tf.gather_nd(CP_locs, CP_idx+[0,1])  # q_k-1
q1 = tf.gather_nd(CP_locs, CP_idx+[0,0])   # q_k
q2 = tf.gather_nd(CP_locs, CP_idx+[1,0])   # q_k+1
q3 = tf.gather_nd(CP_locs, CP_idx+[1,1])  # q_k+2
r = x_input - q1
x_mapped = (
    tf.multiply(q0, -0.5 + tf.multiply( r, 1-0.5*r )) + 
    tf.multiply(q1, 1 + tf.multiply( tf.pow(r, 2), -2.5+1.5*r )) + 
    tf.multiply(tf.multiply(q2, r ), 0.5 + tf.multiply( r, 2-1.5*r )) + 
    tf.multiply(tf.multiply(q3, tf.pow(r,2) ), -0.5+0.5*r )
    ) # [r**3, r**2, r**1, r**0] * A * [q_k-1, q_k, q_k+1, q_k+2].T


## 1D
x_input = tf.Variable([.5,  1.5], dtype=tf.float32)
CP_locs = tf.Variable([-1, 0, 1, 2, 3], dtype=tf.float32)
CP_idx = tf.Variable([[1],[2]]) 
q0 = tf.gather_nd(CP_locs, CP_idx-1)  # q_k-1
q1 = tf.gather_nd(CP_locs, CP_idx)   # q_k
q2 = tf.gather_nd(CP_locs, CP_idx+1)   # q_k+1
q3 = tf.gather_nd(CP_locs, CP_idx+2)  # q_k+2
r = x_input - q1
x_mapped = (
    tf.multiply(q0, -0.5 + tf.multiply( r, 1-0.5*r )) + 
    tf.multiply(q1, 1 + tf.multiply( tf.pow(r, 2), -2.5+1.5*r )) + 
    tf.multiply(tf.multiply(q2, r ), 0.5 + tf.multiply( r, 2-1.5*r )) + 
    tf.multiply(tf.multiply(q3, tf.pow(r,2) ), -0.5+0.5*r )
    ) # [r**3, r**2, r**1, r**0] * A * [q_k-1, q_k, q_k+1, q_k+2].T
print(x_mapped)
print(x_mapped-x_input)


## 2D
x_input = tf.Variable([[.5, 1.5], [1.5, 0.5]], dtype=tf.float32)
x1 = tf.Variable([-1,0,1,2,3], dtype=tf.float32)
CP_locs = tf.stack(tf.meshgrid(x1,x1), axis=2)
CP_idx = tf.Variable([[2,1],[1,2]]) 
q0 = tf.gather_nd(CP_locs, CP_idx+[0,1])  # q_k-1
q1 = tf.gather_nd(CP_locs, CP_idx+[0,0])   # q_k
q2 = tf.gather_nd(CP_locs, CP_idx+[1,0])   # q_k+1
q3 = tf.gather_nd(CP_locs, CP_idx+[1,1])  # q_k+2
r = x_input - q1
x_mapped = (
    tf.multiply(q0, -0.5 + tf.multiply( r, 1-0.5*r )) + 
    tf.multiply(q1, 1 + tf.multiply( tf.pow(r, 2), -2.5+1.5*r )) + 
    tf.multiply(tf.multiply(q2, r ), 0.5 + tf.multiply( r, 2-1.5*r )) + 
    tf.multiply(tf.multiply(q3, tf.pow(r,2) ), -0.5+0.5*r )
    ) # [r**3, r**2, r**1, r**0] * A * [q_k-1, q_k, q_k+1, q_k+2].T
print(x_mapped)
print(x_mapped-x_input)
'''