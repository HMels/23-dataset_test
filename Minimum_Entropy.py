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

#%reload_ext tensorboard

#%% functions

@tf.function
def Rel_entropy(ch1,ch2):
    '''
    Parameters
    ----------
    x_input : float32 array 
        The array containing the [x1, x2] locations of all localizations.

    Returns
    -------
    rel_entropy : float32
        The relative entropy as calculated by Cnossen 2021.

    '''    
    N = ch1.shape[0]    
    expDistances = tf.reduce_sum(
        tf.math.exp( -1 * KL_divergence( ch1, ch2 ) )  / N
         , 0 )
    
    # delete all zero values
    boolean_mask = tf.cast(expDistances, dtype=tf.bool)              
    no_zeros = tf.boolean_mask(expDistances, boolean_mask, axis=0)
    
    return -1*tf.reduce_sum( tf.math.log(
            no_zeros 
            ))


@tf.function
def KL_divergence(ch1, ch2, k = 32):
    '''
    Parameters
    ----------
    mu_i, mu_j : 2D float32 array
        The array containing the [x1, x2] locations of the localizations i and j.
    sigma_i, sigma_j : 2D float32 array
        The array containing the [x1, x2] std of the localizations i and j.
    k : int
        The number of kNN the KL-Divergence should be calculated for

    Returns
    -------
    D_KL : float
        The Kullback Leibler divergence as described by Cnossen 2021.

    '''
    N_locs = ch1.shape[0]
    typical_CRLB = .15*100/10   # CRLB is typically 0.15 pix in size
    sigma2_j = typical_CRLB * tf.ones([k,N_locs,2], dtype = float)

    dist_squared = tf.square(ch1 - ch2)
      
    return 0.5*tf.reduce_sum( dist_squared / sigma2_j**2 , 2)


#%% Classes

class Polynomial_module(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a certain polynomial deformation via the Polynomial Class
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, name=None):
        super().__init__(name=name)
        
        self.polynomial = Polynomial()
    
    @tf.function # to indicate code should run as graph
    def call(self, ch1, ch2):
        ch2_mapped = self.polynomial(ch2)
        return Rel_entropy(ch1, ch2_mapped)
    
    
class Parameterized_module(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, name=None):
        super().__init__(name=name)
        
        self.shift = Shift()
        self.rotation = Rotation()
    
    @tf.function # to indicate code should run as graph
    def call(self, ch1, ch2):
        ch2_mapped = self.rotation(
            self.shift( ch2 )
            )
        return Rel_entropy(ch1, ch2_mapped)


#%% Polynomial Mapping Class
class Polynomial(tf.keras.layers.Layer):
    '''
    Sublayer Polynomial
    ----------
    __init__ : constructs the class together with the initial parameters matrices 
            M1 and M2, in which index [i,j] stands for x1**i * x2**j 
            and start at x1 = x1 and x2 = x2
    ----------
    call : takes input x_input, a Nx2 float32 array containing all localizations
            and transforms them polynomialy using M1 and M2
    '''
     
    def __init__(self, name = None): 
        super().__init__(name=name) 
        self.M1 = tf.Variable([[0.0, 0.0],
                               [1.0, 0.0]],
                              dtype=tf.float32, trainable=True, name = 'M1'
                              )
        self.M2 = tf.Variable([[0.0, 1.0],
                               [0.0, 0.0]],
                              dtype=tf.float32, trainable=True, name = 'M2'
                              )
        
     
    @tf.function
    def call(self, x_input):
        y = tf.zeros(x_input.shape)[None]
        m = 2
        for i in range(m):
            for j in range(m):
                y = tf.concat([y, 
                                [tf.stack([ 
                                   self.M1[i,j] * ( x_input[:,0]**i ) * ( x_input[:,1]**j ),
                                   self.M2[i,j] * ( x_input[:,0]**i ) * ( x_input[:,1]**j )
                                   ], axis = 1)]
                               ], axis = 0)
        
        return tf.reduce_sum(y, axis = 0)


#%% Parametrized Mapping Class
class Shift(tf.keras.layers.Layer):
    '''
    Sublayer Shift
    ----------
    -Constructs shift, initial shift = [0,0]
    '''
    def __init__(self, name = None):
        super().__init__(name=name)
        
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')
        
    @tf.function
    def call(self, x_input):
        return x_input + self.d[None]
    
        
class Rotation(tf.keras.layers.Layer):
    '''
    Sublayer Rotation
    ----------
    -Constructs Rotation, initial rotation = 0 degrees
    '''
    def __init__(self, name = None):
        super().__init__(name=name)
        
        self.theta = tf.Variable(0, dtype=tf.float32, trainable=True, name='rotation')
        
    @tf.function
    def call(self, x_input):
        cos = tf.cos(self.theta * 0.0175/100)
        sin = tf.sin(self.theta * 0.0175/100)
        
        x1 = x_input[:,0]*cos - x_input[:,1]*sin
        x2 = x_input[:,0]*sin + x_input[:,1]*cos
        r = tf.stack([x1, x2], axis =1 )
        return r