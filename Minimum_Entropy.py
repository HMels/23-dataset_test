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

'''
import tensorflow as tf
import ML_functions
from photonpy import PostProcessMethods, Context

#%reload_ext tensorboard

#%% optimization function
def get_apply_grad_fn():
    def apply_grad(ch1, ch2, model, opt):
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

        Returns
        -------
        y : float32
            the relative entropy.

        '''
        i = 0
        y1 = 1000
        step_size = 1
        
        while step_size > 0.01:
            with tf.GradientTape() as tape:
                y = model(ch1, ch2)
                        
            gradients = tape.gradient(y, model.trainable_variables)
                 
            step_size = tf.abs(y1 - y)
            if i%100 == 0:
                print('------------------------ ( i = ', i, ' )------------------------')
                print('- theta = ',model.rotation.theta.numpy(),'shift = ',model.shift.d.numpy())
                print('- Entropy = ',y.numpy(),'gradients = ',gradients)
            
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            
            y1 = y
            i += 1 
        
        return y
    return apply_grad


#%% functions

#@tf.function
def KL_divergence(mu_i, mu_j):
    '''
    Parameters
    ----------
    mu_i, mu_j : 2D float32 array
        The array containing the [x1, x2] locations of the localizations i and j.
    sigma_i, sigma_j : 2D float32 array
        The array containing the [x1, x2] std of the localizations i and j.

    Returns
    -------
    D_KL : float
        The Kullback Leibler divergence as described by Cnossen 2021.

    '''
    typical_CRLB = .15*100/10   # CRLB is typically 0.15 pix in size
    K = mu_i.shape[1]
    sigma2_i = typical_CRLB * tf.ones(K, dtype = float)
    sigma2_j = typical_CRLB * tf.ones(K, dtype = float)

    return tf.reduce_sum( (mu_i - mu_j)**2 / sigma2_j[None], 1 )

    """
    D_KL = -K/2
    for k in range(K):
        D_KL += ( (1/2) * tf.math.log(sigma2_j[k] / sigma2_i[k]) 
                 + sigma2_i[k] / sigma2_j[k] 
                 + (1/sigma2_j[k]) * (mu_i[k] - mu_j[k])**2
                 )
    return D_KL
    """

#@tf.function
def Rel_entropy1(ch1, ch2):
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
    nn_ch1, nn_ch2 = ML_functions.vicinity_neighbours_numpy(ch1, ch2, 3)
    
    N = ch1.shape[1]
    rel_entropy = 0
    j = 0
    for i in range(N):
        temp = 0
        while nn_ch1[j] == i:
            # Calculate KL-Divergence between loc_i in ch1 and its nearest neighbours in ch2
            D_KL = KL_divergence(ch1[:,i], ch2[:, int(nn_ch2[j].numpy() ) ]) 
            temp += tf.math.exp( - D_KL )
            j += 1                      # the next index

        if temp != 0:                   # ignore empty values
            rel_entropy += tf.math.log( (1/N) * temp  )
    return -1.0 * rel_entropy / N


#@tf.function
def Rel_entropy(ch1, ch2):
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
    N = ch1.shape[1]

    return tf.reduce_sum(KL_divergence(ch1, ch2))

    """
    rel_entropy = 0.0
    for i in range(N):
        # Calculate KL-Divergence between loc_i in ch1 and its nearest neighbours in ch2
        D_KL = KL_divergence( ch1[:, i], ch2[:, i] )
        temp = tf.math.exp( - D_KL )

        if temp != 0.0:                   # ignore empty values
            rel_entropy += tf.math.log( (1/N) * temp  )
    return -1.0 * rel_entropy / N
    """


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
    
    @tf.function # to indicate code should run as graph
    def call(self, ch1, ch2):
        #ch2_logits = self.polynomial(ch2)
        ch2_logits = self.rotation(
            self.shift( ch2 )
            )
        
        y = Rel_entropy(ch1, ch2_logits)
        return y

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
        
        
    @tf.function
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
        
        self.d = tf.Variable([0, 0], dtype=tf.float32, trainable=True, name='shift')
        
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
        x1 = x_input[:,0]*tf.math.cos(self.theta) - x_input[:,1]*tf.math.sin(self.theta)
        x2 = x_input[:,0]*tf.math.sin(self.theta) + x_input[:,1]*tf.math.cos(self.theta)
        r = tf.stack([x1, x2], axis =1)
        return r
        