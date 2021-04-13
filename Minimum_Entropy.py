# Minimum_Entropy.py
'''
This script is used to calculate the Mapping via the Minimum Entropy Method described by Cnossen2021

The script contains the next functions:
- KL_divergence() 
    the Kullback-Leibler divergence between localization i and j
- Rel_entropy()
    the relative entropy for certain localizations
    

The script also contains the next Model in the form of Classes
- PolMod:           the main model calculating the minimum entropy
|- Polynomial
|- Shift
|- Rotation

'''
import tensorflow as tf
import ML_functions
#from photonpy import PostProcessMethods, Context

#%reload_ext tensorboard

#%% optimization function
def get_apply_grad_fn():
    def apply_grad(ch1, ch2, model, opt, error = 0.01):
        '''
        The function that minimizes a certain model using TensorFlow GradientTape()
        
        Parameters
        ----------
        x_input : Nx2 float32 array
            Contains the [x1, x2] locs of all localizations.
        model : TensorFlow Keras Model
            The model that needs to be optimized. In our case, This model will be
            PolMod() which contains the sublayer Polynomial().
        opt : TensorFlow Keras Optimizer 
            The optimizer which our function uses for Optimization. In our case
            this will be a TensorFlow.Optimizer.Adam().
        error : float
            The acceptable error in Entropy

        Returns
        -------
        y : float32
            the relative entropy.

        '''
        i = 0
        y1 = 1000
        step_size = 1
        
        while step_size > error:
            with tf.GradientTape() as tape:
                y = model(ch1, ch2)
                        
            gradients = tape.gradient(y, model.trainable_variables)
                 
            step_size = tf.abs(y1 - y)
            if i%100 == 0:
                print('------------------------ ( i = ', i, ' )------------------------')
                print('- rotation = ',model.rotation.theta.numpy()*180/3.14,'shift = ',model.shift.d.numpy())
                print('- Entropy = ',y.numpy())
          
            
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            
            y1 = y
            i += 1 
        
        return y
    return apply_grad


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


def KL_divergence(ch1, ch2, k = 8):
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

    dist_squared = ML_functions.KNN(ch1,ch2, k)
      
    return 0.5*tf.reduce_sum( dist_squared / sigma2_j**2 , 2)
#%% Classes

class PolMod(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a certain polynomial deformation via the Polynomial Class
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, name=None):
        super().__init__(name=name)
        
        #self.polynomial = Polynomial()
        self.shift = Shift()
        self.rotation = Rotation()
    
    #@tf.function # to indicate code should run as graph
    def call(self, ch1, ch2):
        #ch2_mapped = self.polynomial(ch2)
        ch2_mapped = self.rotation(
            self.shift( ch2 )
            )
        
        return Rel_entropy(ch1, ch2_mapped)


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
                               [1.0, 0.0] ],
                              dtype=tf.float32, trainable=True, name = 'M1'
                              )
        self.M2 = tf.Variable([[0.0, 1.0],
                               [0.0, 0.0]],
                              dtype=tf.float32, trainable=True, name = 'M2'
                              )
        
        
    #@tf.function
    def call(self, x_input):
        y = tf.zeros(x_input.shape)[None]
        for i in range(2):
            for j in range(2):
                y = tf.concat([y, 
                                [tf.transpose(tf.stack([ 
                                   self.M1[i,j] * ( x_input[0,:]**i ) * ( x_input[1,:]**j ),
                                   self.M2[i,j] * ( x_input[0,:]**i ) * ( x_input[1,:]**j )
                                   ], axis = 1))]
                               ], axis = 0)
        
        return tf.reduce_sum(y, axis = 0)


#%% new classes
class Shift(tf.keras.layers.Layer):
    '''
    Sublayer Shift
    ----------
    -Constructs shift, initial shift = [0,0]
    '''
    def __init__(self, name = None):
        super().__init__(name=name)
        
        self.d = tf.Variable([0.0, 0.0], dtype=tf.float32, trainable=True, name='shift')
        
    #@tf.function
    def call(self, x_input):
        return x_input - self.d[None]
    
        
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
        x1 = x_input[:,0]*tf.math.cos(self.theta) + x_input[:,1]*tf.math.sin(self.theta)
        x2 = -1* x_input[:,0]*tf.math.sin(self.theta) + x_input[:,1]*tf.math.cos(self.theta)
        r = tf.stack([x1, x2], axis =1)
        return r
        