# Minimum_Entropy_eager.py
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

- Parameterized_module_complex:      the Parameterized model for calculating the minimum entropy
|- Shift
|- Rotation
|- Shear
|- Scaling

- Polynomial_module:                the Polynomial model for calculating the minimum entropy
|- Polynomial
'''
import tensorflow as tf
import numpy as np

#%reload_ext tensorboard

#%% functions

#@tf.function
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


#@tf.function
def KL_divergence(ch1, ch2, k = 16):
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

    dist_squared = KNN(ch1,ch2, k)
      
    return 0.5*tf.reduce_sum( dist_squared / sigma2_j**2 , 2)


#%% KNN
#@tf.function
def KNN(ch1, ch2, k):
    '''
    k-Nearest Neighbour Distance calculator

    Parameters
    ----------
    ch1, ch2 : 2D float32 array
        The array containing the [x1, x2] locations of the localizations i and j.
    sigma_i, sigma_j : 2D float32 array
        The array containing the [x1, x2] std of the localizations i and j.
    k : int
        The number of kNN the KL-Divergence should be calculated for

    Returns
    -------
    knn : [k, N, 2] TensorFlow Tensor
        Tensor Containing the squared [x1,x2] distances for k rows of kNN, 
        for all N localizations in the colums.

    '''
    N_locs = ch1.shape[0]
    # distances contains all distances with the second axis being the distances to ch1
    # and the third axis being the [x1,x2] axis
    
    distances = tf.square(tf.add( 
        ch1[None,:,:] , tf.negative( ch2[:,None,:] )
        ))
    abs_distances = tf.reduce_sum(distances, 2)
    
    neg_one = tf.constant(-1.0, dtype=tf.float32)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply( tf.transpose(abs_distances) , neg_one)
    # get the indices
    _, indx = tf.nn.top_k(neg_distances, k)
        
    # getting index in the right format for a gather_nd
    indx = tf.reshape(indx,[1, indx.shape[0]*indx.shape[1] ])
    indx1 = np.linspace(0,  N_locs-1, N_locs, dtype = int) * np.ones([k,1], dtype = int)
    indx1 = tf.reshape(indx1, [1, indx.shape[0]*indx.shape[1] ])
    indx = tf.transpose( tf.stack( [indx, indx1] ) )
    
    knn = tf.reshape( tf.gather_nd( distances, indx ), [k, N_locs, 2] )
    return knn


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
    
    #@tf.function # to indicate code should run as graph
    def call(self, ch1, ch2):
        ch2_mapped = self.polynomial(ch2)
        return Rel_entropy(ch1, ch2_mapped)
    
    
class Parameterized_module_simple(tf.keras.Model):
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
    
    #@tf.function # to indicate code should run as graph
    def call(self, ch1, ch2):
        ch2_mapped = self.rotation(
            self.shift( ch2 )
            )
        return Rel_entropy(ch1, ch2_mapped)


class Parameterized_module_complex(tf.keras.Model):
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
        self.shear = Shear()
        self.scaling = Scaling()
    
    #@tf.function # to indicate code should run as graph
    def call(self, ch1, ch2):
        ch2_mapped = self.scaling( self.shear(
            self.rotation( self.shift( ch2 ) )
            ) )
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
        self.m = 2
        
        
    #@tf.function
    def call(self, x_input):
        y = tf.zeros(x_input.shape)[None]
        for i in range(self.m):
            for j in range(self.m):
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
        
        self.d = tf.Variable([-40, -30], dtype=tf.float32, trainable=True, name='shift')
        
    #@tf.function
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
        
    #@tf.function
    def call(self, x_input):
        x1 = x_input[:,0]*tf.math.cos(self.theta) - x_input[:,1]*tf.math.sin(self.theta)
        x2 = x_input[:,0]*tf.math.sin(self.theta) + x_input[:,1]*tf.math.cos(self.theta)
        r = tf.stack([x1, x2], axis =1 )
        return r
    
    
class Shear(tf.keras.layers.Layer):
    '''
    Sublayer Shear
    ----------
    -Constructs Shear, initial shear = [0,0]
    '''
    def __init__(self, name = None):
        super().__init__(name=name)
        
        self.shear = tf.Variable([0,0], dtype=tf.float32, trainable=False, name='shear')
        
    #@tf.function
    def call(self, x_input):
        x1 = x_input[:,0] + self.shear[0] * x_input[:,1]
        x2 = self.shear[1] * x_input[:,0] + x_input[:,1]
        r = tf.stack([x1, x2], axis = 1 )
        return r
    

class Scaling(tf.keras.layers.Layer):
    '''
    Sublayer Scaling
    ----------
    -Constructs Scaling, initial scaling = [1,1]
    '''
    def __init__(self, name = None):
        super().__init__(name=name)
        
        self.scaling = tf.Variable([1,1], dtype=tf.float32, trainable=False, name='Scaling')
        
    #@tf.function
    def call(self, x_input):
        x1 = self.scaling[0] * x_input[:,0]
        x2 = self.scaling[1] * x_input[:,1]
        r = tf.stack([x1, x2], axis = 1 )
        return r
    
