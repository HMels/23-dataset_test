
# MinEntropy.py
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

#%reload_ext tensorboard

#%% functions

@tf.autograph.experimental.do_not_convert
def Rel_entropy(ch1,ch2):#,neighbour_idx):
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
    
    '''
        i need to try to make a convert to Neighbour_format function that takes 
        neighbours and places everything in a 700x365x2 format of ch1 and ch2.
        
        The program will than 
        -reduce_sum(axis=2) in DKL to add the dimensions
        -reduce_sum(axis=1) in Entropy to add the exponents
        -reduce_sum(axis=0) in Entropy to add the log
    '''
    
    N = ch1.shape[0]
    '''
    DKL = tf.Variable([], dtype = tf.float32)
    for neighbour in neighbour_idx:
        
        ch1_temp = tf.gather_nd(ch1, neighbour[0,:][:,None])
        ch2_temp = tf.gather_nd(ch2, neighbour[1,:][:,None])
        
        D_KL = tf.concat([ DKL,  [ tf.math.log( 
            tf.reduce_sum( tf.math.exp( 
                -1 * KL_divergence( ch1_temp, ch2_temp ) 
                )  / N )
            ) ] ], axis = 0)
    '''
    return -1*tf.reduce_sum(
        tf.math.exp(
            -1*KL_divergence(ch1, ch2) / N )
        )#-1*tf.reduce_sum( D_KL )


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
    return 0.5*tf.reduce_sum( dist_squared / typical_CRLB**2 , 1)


#%% Classes  

class Poly3Mod(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a certain polynomial deformation via the Polynomial Class
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, name = None): 
        super().__init__(name=name) 
        self.M1 = tf.Variable([[0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              dtype=tf.float32, trainable=True, name = 'M1'
                              )
        self.M2 = tf.Variable([[0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              dtype=tf.float32, trainable=True, name = 'M2'
                              )
    
    @tf.function 
    def call(self, ch1, ch2):#, neighbour_idx):
        ch2_mapped = self.transform(ch2)
        return Rel_entropy(ch1, ch2_mapped)#, neighbour_idx)
    
    @tf.autograph.experimental.do_not_convert
    def transform(self, x_input):
        y = tf.stack([
            tf.concat([self.M1[0,0]*tf.ones([x_input.shape[0],1]), 
                       self.M1[1,0]*x_input[:,0][:,None],
                       self.M1[0,1]*x_input[:,1][:,None],
                       self.M1[1,1]*(x_input[:,0]*x_input[:,1])[:,None],
                       self.M1[2,1]*((x_input[:,0]**2)*x_input[:,1])[:,None],
                       self.M1[2,2]*((x_input[:,0]*x_input[:,1])**2)[:,None],
                       self.M1[1,2]*(x_input[:,0]*(x_input[:,1]**2))[:,None],
                       self.M1[0,2]*(x_input[:,1]**2)[:,None],
                       self.M1[2,0]*(x_input[:,0]**2)[:,None]
                       ], axis = 1),
            tf.concat([self.M2[0,0]*tf.ones([x_input.shape[0],1]), 
                       self.M2[1,0]*x_input[:,0][:,None],
                       self.M2[0,1]*x_input[:,1][:,None],
                       self.M2[1,1]*(x_input[:,0]*x_input[:,1])[:,None],
                       self.M2[2,1]*((x_input[:,0]**2)*x_input[:,1])[:,None],
                       self.M2[2,2]*((x_input[:,0]*x_input[:,1])**2)[:,None],
                       self.M2[1,2]*(x_input[:,0]*(x_input[:,1]**2))[:,None],
                       self.M2[0,2]*(x_input[:,1]**2)[:,None],
                       self.M2[2,0]*(x_input[:,0]**2)[:,None]
                       ], axis = 1),
            ], axis = 2)
        return tf.reduce_sum(y, axis = 1)


#%%
class ShiftMod(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, name=None):
        super().__init__(name=name)
        
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')

    @tf.function
    def call(self, ch1, ch2):
        ch2_mapped = self.transform( ch2 )
        return Rel_entropy(ch1, ch2_mapped)
    
    @tf.autograph.experimental.do_not_convert
    def transform(self, x_input):
        return x_input + self.d[None]
    
    
#%%
class RotationMod(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, name=None):
        super().__init__(name=name)
        
        self.theta = tf.Variable(0, dtype=tf.float32, trainable=True, name='rotation')
        
    @tf.function
    def call(self, ch1, ch2):
        ch2_mapped = self.transform( ch2 )
        return Rel_entropy(ch1, ch2_mapped)
    
    @tf.autograph.experimental.do_not_convert
    def transform(self, x_input):
        cos = tf.cos(self.theta * 0.0175/100)
        sin = tf.sin(self.theta * 0.0175/100)
        
        x1 = x_input[:,0]*cos - x_input[:,1]*sin
        x2 = x_input[:,0]*sin + x_input[:,1]*cos
        r = tf.stack([x1, x2], axis =1 )
        return r
