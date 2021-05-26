
# Minimum_Entropy.py
'''
This script is used to calculate the Mapping via the Minimum Entropy Method described by Cnossen2021

The script contains the next functions:
- KL_divergence() 
    the Kullback-Leibler divergence between localization i and j
- Rel_entropy()
    the relative entropy for certain localizations
    

The script also contains the next Model in the form of Classes
- Parameterized_module_simple:      the Parameterized model for calculating the minimum entropy
|- Shift
|- Rotation


- Polynomial_module:                the Polynomial model for calculating the minimum entropy
|- Polynomial
'''
import tensorflow as tf


#%%
#@tf.autograph.experimental.do_not_convert
@tf.function
def Rel_entropy1(ch1, ch2):
    return tf.reduce_sum(tf.square(ch1-ch2))


#%% functions
@tf.autograph.experimental.do_not_convert
def Rel_entropy(ch1,ch2):
    '''
    Parameters
    ----------
    ch1, ch2 : float32 array 
        The array containing the [x1, x2] locations of all localizations.
    idxlist : list
        List containing per indice of ch1 the neighbours in ch2.

    Returns
    -------
    rel_entropy : float32
        The relative entropy as calculated by Cnossen 2021.

    ''' 
    N = ch1.shape[0]
    expDist = tf.reduce_sum( tf.math.exp(
            -1*KL_divergence(ch1, ch2) / N ) / N
        , axis = 1)
    return -1*tf.reduce_sum(tf.math.log( 
        expDist
        ))


@tf.autograph.experimental.do_not_convert
def KL_divergence(ch1, ch2):
    '''
    Parameters
    ----------
    ch1, ch2 : 2D float32 array
        The array containing the [x1, x2] locations of the localizations i and j.

    Returns
    -------
    D_KL : float array
        The Kullback Leibler divergence as described by Cnossen 2021.

    '''
    typical_CRLB = 15   # CRLB is typically 0.15 pix in size
    dist_squared = tf.square(ch1 - ch2)
    return 0.5*tf.reduce_sum( dist_squared / typical_CRLB**2 , 2)

   
#%% ShiftRotMod
class ShiftRotMod(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    
    Parameters
    ----------
    d : tf.float32
        The amount of shift in nm
    theta : tf.float32
        The amount of rotation in 0.01 degrees
        
    Returns
    ----------
    Rel_entropy : tf.float32
        The relative entropy of the current mapping
        
    '''
    def __init__(self, name='shift'):
        super().__init__(name=name)
        
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')
        self.theta = tf.Variable(0, dtype=tf.float32, trainable=True, name='rotation')
        

    @tf.function
    def call(self, ch1, ch2):
        ch2_mapped =self.transform_mat(ch2)
        return Rel_entropy(ch1, ch2_mapped)
    
    
    @tf.function
    def transform_mat(self, ch2):
        ch2 = self.shift_transform_mat( ch2 )
        return self.rot_transform_mat( ch2 )
    
    
    @tf.function
    def transform_vec(self, ch2):
        ch2 = self.shift_transform_vec( ch2 )
        return self.rot_transform_vec( ch2 )
    
    
    @tf.function
    def shift_transform_vec(self, x_input):
        return x_input + self.d[None]
    
    
    @tf.function
    def shift_transform_mat(self, x_input):
        return x_input + self.d[None,None]
    
    
    @tf.function
    def rot_transform_vec(self, x_input):
        cos = tf.cos(self.theta/100 * 0.0175)
        sin = tf.sin(self.theta/100 * 0.0175)
        
        x1 = x_input[:,0]*cos - x_input[:,1]*sin
        x2 = x_input[:,0]*sin + x_input[:,1]*cos
        r = tf.stack([x1, x2], axis =1 )
        return r
    
    
    @tf.function
    def rot_transform_mat(self, x_input):
        cos = tf.cos(self.theta/100 * 0.0175)
        sin = tf.sin(self.theta/100 * 0.0175)
        
        x1 = x_input[:,:,0]*cos - x_input[:,:,1]*sin
        x2 = x_input[:,:,0]*sin + x_input[:,:,1]*cos
        r = tf.stack([x1, x2], axis =2 )
        return r

    
    