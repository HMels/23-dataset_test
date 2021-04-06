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
import tensorflow.math as mth
import ML_functions
from photonpy import PostProcessMethods, Context

#%reload_ext tensorboard

#%% optimization function
def get_apply_grad_fn():
    #@tf.function
    def apply_grad(ch1, ch2, model, opt):
        '''
        The function that minimizes a certain model using TensorFlow GradientTape()
        
        Parameters
        ----------
        x_input : 2xN float32 array
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
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            y = model(ch1, ch2)
            
        gradients = tape.gradient(y, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        
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
    K = mu_i.shape[0]
    sigma2_i = typical_CRLB * tf.ones(K, dtype = float)
    sigma2_j = typical_CRLB * tf.ones(K, dtype = float)
    
    D_KL = -K/2
    for k in range(K):
        D_KL += ( (1/2) * mth.log(sigma2_j[k] / sigma2_i[k]) 
                 + sigma2_i[k] / sigma2_j[k] 
                 + (1/sigma2_j[k]) * (mu_i[k] - mu_j[k])**2
                 )
    return D_KL
    

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

    We start with a simple model, in which we only take the nearest neighbour of
    localization i, so for this case there is no need for a loop over j
    '''
    nn_ch1, nn_ch2 = ML_functions.vicinity_neighbours(ch1, ch2, 4)
    
    N = ch1.shape[1]
    rel_entropy = 0
    j = 0
    for i in range(N):
        temp = 0
        while nn_ch1[j] == i:
            # Calculate KL-Divergence between loc_i in ch1 and its nearest neighbours in ch2
            D_KL = KL_divergence(ch1[:,i], ch2[:, int(nn_ch2[j].numpy() ) ]) 
            temp += mth.exp( - D_KL )
            j += 1
        rel_entropy += mth.log( (1/N) * temp  )
            
    return -1 * rel_entropy / N


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
        
        self.polynomial = Polynomial()
    
   # @tf.function # to indicate code should run as graph
    def call(self, ch1, ch2):
        ch2_logits = self.polynomial(ch2)
        return Rel_entropy(ch1, ch2_logits)

class Polynomial(tf.keras.layers.Layer):
    '''
    Sublayer Polynomial
    ----------
    __init__ : constructs the class together with the initial parameters matrices 
            M1 and M2, in which index [i,j] stands for x1**i * x2**j 
            and start at x1 = x1 and x2 = x2
    ----------
    call : takes input x_input, a 2xN float32 array containing all localizations
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
