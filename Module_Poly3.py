# Poly3Mod.py

import tensorflow as tf

import generate_neighbours
import train_model
from MinEntropy_fn import Rel_entropy


#%% run_optimization
def run_optimization(ch1, ch2, maxDistance=50, threshold=10, learning_rate=1, direct=False):
    '''
    Parameters
    ----------
    ch1 ,ch2 : Nx2 Tensor
        Tensor containing the localizations of their channels.
    maxDistance : float32, optional
        The distance in which the Nearest Neighbours will be searched. 
        The default is 50nm.
    threshold : int, optional
        The amount of rejections before ending the optimization loop.
        The default is 10.
    learning_rate : float, optional
        The initial learning rate of our optimizer. The default is 1.
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.

    Returns
    -------
    mods : Models() Class
        Class containing the information and functions of the optimization models.
    ch2 : Nx2 float Tensor
        Mapped version of ch2

    '''
    
    if direct:          # Direct
        nn1=None
        nn2=None
        model=Poly3Mo_direct()
        
    else:               # Generate Neighbours
        neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
        ch1.numpy(), ch2.numpy(), maxDistance=maxDistance, threshold=None)
        #nn1 = ch1[:,None]
        nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
        #nn2 = ch2[:,None]
        nn2 = tf.Variable( neighbours_B, dtype = tf.float32)
        model=Poly3Mod()
        
    # The Model
    mods = train_model.Models(model=model, learning_rate=learning_rate, 
                  opt=tf.optimizers.Adagrad, threshold=threshold)
    
    ## Training Loop
    model_apply_grads = train_model.get_apply_grad_fn()
    return model_apply_grads(ch1, ch2, mods, nn1, nn2)


#%% Poly3Mod  
class Poly3Mod(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a certain polynomial deformation via the Polynomial Class
    - calculates the relative entropy via Rel_entropy()    
    '''
    
    def __init__(self,name = 'polynomial'): 
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
    def call(self, ch1, ch2):
        ch2_mapped = self.transform_mat(ch2)
        return Rel_entropy(ch1, ch2_mapped)
    
    @tf.function
    def transform_vec(self, x_input):
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
    
    
    @tf.function
    def transform_mat(self, x_input):
        y = tf.stack([
            tf.concat([self.M1[0,0]*tf.ones([1, x_input.shape[0], x_input.shape[1]]), 
                       self.M1[1,0]*x_input[:,:,0][None],
                       self.M1[0,1]*x_input[:,:,1][None],
                       self.M1[1,1]*(x_input[:,:,0]*x_input[:,:,1])[None],
                       self.M1[2,1]*((x_input[:,:,0]**2)*x_input[:,:,1])[None],
                       self.M1[2,2]*((x_input[:,:,0]*x_input[:,:,1])**2)[None],
                       self.M1[1,2]*(x_input[:,:,0]*(x_input[:,:,1]**2))[None],
                       self.M1[0,2]*(x_input[:,:,1]**2)[None],
                       self.M1[2,0]*(x_input[:,:,0]**2)[None]
                       ], axis = 0)[:,:,:,None],
            tf.concat([self.M2[0,0]*tf.ones([1, x_input.shape[0], x_input.shape[1]]), 
                       self.M2[1,0]*x_input[:,:,0][None],
                       self.M2[0,1]*x_input[:,:,1][None],
                       self.M2[1,1]*(x_input[:,:,0]*x_input[:,:,1])[None],
                       self.M2[2,1]*((x_input[:,:,0]**2)*x_input[:,:,1])[None],
                       self.M2[2,2]*((x_input[:,:,0]*x_input[:,:,1])**2)[None],
                       self.M2[1,2]*(x_input[:,:,0]*(x_input[:,:,1]**2))[None],
                       self.M2[0,2]*(x_input[:,:,1]**2)[None],
                       self.M2[2,0]*(x_input[:,:,0]**2)[None]
                       ], axis = 0)[:,:,:,None]
            ], axis = 3)
        return tf.reduce_sum(tf.reduce_sum(y, axis = 0), axis = 3)
    
    
#%% Poly3Mo_direct
class Poly3Mo_direct(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a certain polynomial deformation via the Polynomial Class
    - calculates the relative entropy via Rel_entropy()    
    '''
    
    def __init__(self, name = 'polynomial'): 
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
    def call(self, ch1, ch2):
        ch2_mapped = self.transform_vec(ch2)
        return tf.reduce_sum(tf.square(ch1-ch2_mapped))    
    
    @tf.function
    def transform_vec(self, x_input):
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
    
    
    @tf.function
    def transform_mat(self, x_input):
        y = tf.stack([
            tf.concat([self.M1[0,0]*tf.ones([1, x_input.shape[0], x_input.shape[1]]), 
                       self.M1[1,0]*x_input[:,:,0][None],
                       self.M1[0,1]*x_input[:,:,1][None],
                       self.M1[1,1]*(x_input[:,:,0]*x_input[:,:,1])[None],
                       self.M1[2,1]*((x_input[:,:,0]**2)*x_input[:,:,1])[None],
                       self.M1[2,2]*((x_input[:,:,0]*x_input[:,:,1])**2)[None],
                       self.M1[1,2]*(x_input[:,:,0]*(x_input[:,:,1]**2))[None],
                       self.M1[0,2]*(x_input[:,:,1]**2)[None],
                       self.M1[2,0]*(x_input[:,:,0]**2)[None]
                       ], axis = 0)[:,:,:,None],
            tf.concat([self.M2[0,0]*tf.ones([1, x_input.shape[0], x_input.shape[1]]), 
                       self.M2[1,0]*x_input[:,:,0][None],
                       self.M2[0,1]*x_input[:,:,1][None],
                       self.M2[1,1]*(x_input[:,:,0]*x_input[:,:,1])[None],
                       self.M2[2,1]*((x_input[:,:,0]**2)*x_input[:,:,1])[None],
                       self.M2[2,2]*((x_input[:,:,0]*x_input[:,:,1])**2)[None],
                       self.M2[1,2]*(x_input[:,:,0]*(x_input[:,:,1]**2))[None],
                       self.M2[0,2]*(x_input[:,:,1]**2)[None],
                       self.M2[2,0]*(x_input[:,:,0]**2)[None]
                       ], axis = 0)[:,:,:,None]
            ], axis = 3)
        return tf.reduce_sum(tf.reduce_sum(y, axis = 0), axis = 3)
    
    

