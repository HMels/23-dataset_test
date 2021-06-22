# ShiftRotMod.py

import tensorflow as tf

import MinEntropyModules.generate_neighbours as generate_neighbours
import MinEntropyModules.train_model as train_model
from MinEntropyModules.MinEntropy_fn import Rel_entropy


#%% run_optimization
def run_optimization(ch1, ch2, N_it=200, maxDistance=50, learning_rate=1,
                     direct=False, opt=tf.optimizers.Adagrad):
    '''
    Parameters
    ----------
    ch1 ,ch2 : Nx2 Tensor
        Tensor containing the localizations of their channels.
    N_it : int, optional
        Number of iterations used in the training loop. The default is 200
    maxDistance : float32, optional
        The distance in which the Nearest Neighbours will be searched. 
        The default is 50nm.
    learning_rate : float, optional
        The initial learning rate of our optimizer. The default is 1.
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.
    opt : tf.optimizers, optional
        The Optimizer used. The default is tf.optimizers.Adagrad

    Returns
    -------
    mods : Models() Class
        Class containing the information and functions of the optimization models.
    ch2 : Nx2 float Tensor
        Mapped version of ch2

    '''
    
    if direct:          # Generate Neighbours
        nn1=None
        nn2=None
        model=ShiftRotMod_direct()

    else:               # Direct
        neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
        ch1.numpy(), ch2.numpy(), maxDistance=maxDistance, threshold=None)
        #nn1 = ch1[:,None]
        nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
        #nn2 = ch2[:,None]
        nn2 = tf.Variable( neighbours_B, dtype = tf.float32)
        model=ShiftRotMod()

        
    
    # The Model
    mods = train_model.Models(model=model, learning_rate=learning_rate, 
                  opt=opt)
    
    ## Training Loop
    model_apply_grads = train_model.get_apply_grad_fn()
    return model_apply_grads(ch1, ch2, N_it, mods, nn1, nn2)


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
        ch2_mapped = self.transform_mat(ch2)
        return Rel_entropy(ch1, ch2_mapped)
    
    
    @tf.function
    def transform_mat(self, ch2):
        ## Shift
        ch2_mapped = ch2 + self.d[None,None] 
        
        ## Rotate
        cos = tf.cos(self.theta/100 * 0.0175)
        sin = tf.sin(self.theta/100 * 0.0175)
        
        x1 = ch2_mapped[:,:,0]*cos - ch2_mapped[:,:,1]*sin
        x2 = ch2_mapped[:,:,0]*sin + ch2_mapped[:,:,1]*cos
        return tf.stack([x1, x2], axis =2 )
    
    
    @tf.function
    def transform_vec(self, ch2):
        ## Shift
        ch_mapped = ch2 + self.d[None]
        
        ## Rotate
        cos = tf.cos(self.theta/100 * 0.0175)
        sin = tf.sin(self.theta/100 * 0.0175)
        
        x1 = ch_mapped[:,0]*cos - ch_mapped[:,1]*sin
        x2 = ch_mapped[:,0]*sin + ch_mapped[:,1]*cos
        return tf.stack([x1, x2], axis =1 )
    
    
#%% ShiftRotMod_direct
class ShiftRotMod_direct(tf.keras.Model):
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
        ch2_mapped = self.transform_vec(ch2)
        return tf.reduce_sum(tf.square(ch1-ch2_mapped)) 
    
    
    @tf.function
    def transform_mat(self, ch2):
        ## Shift
        ch2_mapped = ch2 + self.d[None,None] 
        
        ## Rotate
        cos = tf.cos(self.theta/100 * 0.0175)
        sin = tf.sin(self.theta/100 * 0.0175)
        
        x1 = ch2_mapped[:,:,0]*cos - ch2_mapped[:,:,1]*sin
        x2 = ch2_mapped[:,:,0]*sin + ch2_mapped[:,:,1]*cos
        return tf.stack([x1, x2], axis =2 )
    
    
    @tf.function
    def transform_vec(self, ch2):
        ## Shift
        ch_mapped = ch2 + self.d[None]
        
        ## Rotate
        cos = tf.cos(self.theta/100 * 0.0175)
        sin = tf.sin(self.theta/100 * 0.0175)
        
        x1 = ch_mapped[:,0]*cos - ch_mapped[:,1]*sin
        x2 = ch_mapped[:,0]*sin + ch_mapped[:,1]*cos
        return tf.stack([x1, x2], axis =1 )