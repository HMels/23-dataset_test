# catmull_rom_splines

import tensorflow as tf

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
#%%

gridsize = 50               # nm, size between controlpoints
x_grid = tf.range(tf.reduce_min(ch2[:,0])-gridsize/2, tf.reduce_max(ch2[:,0])+gridsize*1.5, gridsize)
y_grid = tf.range(tf.reduce_min(ch2[:,1])-gridsize/2, tf.reduce_max(ch2[:,1])+gridsize*1.5, gridsize)
ControlPoints = tf.meshgrid(x_grid, y_grid  )


#%% CatmullRomSplines
class CatmullRomSplines(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, ControlPoints, name='CatmullRomSplines'):
        super().__init__(name=name)
        
        self.A = tf.Variable([
            [-.5, 1.5, -1.5, 0.5],
            [1, -2.5, 2, -.5],
            [-.5, 0, -.5, 0],
            [0, 1, 0, 0]
            ], trainable=False, dtype=tf.float32)
        self.ControlPoints = tf.Variable(ControlPoints, dtype=tf.float32,
                                         trainable=True, name='ControlPoints')


    @tf.function
    def call(self, ch1, ch2):
        #ch2_mapped = self.transform_mat( ch2 )
        ch2_mapped = self.transform_vec( ch2 )
        return Rel_entropy(ch1, ch2_mapped)
    
    
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        r_mat = self.DistMat_vec(x_input)
        x_mapped = tf.matmul(r_mat, tf.matmul(self.A, self.ControlPoints) )
        return x_mapped
    
    
    @tf.autograph.experimental.do_not_convert
    def transform_mat(self, x_input):
        r_mat = self.DistMat_mat(x_input)
        x_mapped = tf.matmul(r_mat, tf.matmul(self.A, self.ControlPoints) )
        return x_mapped
    
    
    @tf.autograph.experimental.do_not_convert
    def DistMat_vec(self, r):
        return tf.Variable([r**3, r**2, r, tf.ones(r.shape, dtype=tf.float32)])
    
    
    @tf.autograph.experimental.do_not_convert
    def DistMat_mat(self, r):
        return tf.Variable([r**3, r**2, r, tf.ones(r.shape, dtype=tf.float32)])