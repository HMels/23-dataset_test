# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:44:37 2021

@author: Mels
"""
# run_optimizationtestversion.py
"""
Created on Wed Apr 14 15:40:25 2021

@author: Mels
"""
import tensorflow as tf
#import Minimum_Entropy
import Minimum_Entropy_eager as Minimum_Entropy
import numpy as np
import dataset_manipulation

#%% initialize function
def initialize_optimizer1(locs_A, locs_B, Map_opt='Parameterized_simple' , Batch_on=False, batch_size=None,
                         num_batches=None, learning_rate=.001, epochs = 50):
     
    ## Shift it 
    model = Minimum_Entropy.Parameterized_module_simple()
    model.shift.trainable = True
    model.rotation.trainable = False
    
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)
    
    model_apply_grads = get_apply_grad_fn_nobatch()
    
    opt = tf.optimizers.Adam(learning_rate=1)
    loss = model_apply_grads(ch1, ch2, model, opt, epochs)
    
    ## Poly it
    ch2_shift = tf.Variable( dataset_manipulation.simple_translation( 
        locs_B, model.shift.d.numpy() , 0 )
        , dtype = tf.float32)
    model.shift.trainable = False
    model.rotation.trainable = True

    
    model_apply_grads = get_apply_grad_fn_nobatch()
    
    opt = tf.optimizers.Adam(learning_rate=0.05)
    loss = model_apply_grads(ch1, ch2_shift, model, opt, epochs)
    
    return model, loss


def initialize_optimizer(locs_A, locs_B, Map_opt='Parameterized_simple' , Batch_on=False, batch_size=None,
                         num_batches=None, learning_rate=.001, epochs = 50):
     
    ## Shift it 
    model = Minimum_Entropy.Parameterized_module_simple()
    model.shift.trainable = True
    model.rotation.trainable = False
    
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)
    
    model_apply_grads = get_apply_grad_fn_nobatch()
    
    opt = tf.optimizers.Adam(learning_rate=1)
    loss = model_apply_grads(ch1, ch2, model, opt, epochs)
    
    ## Poly it
    ch2_shift = tf.Variable( dataset_manipulation.simple_translation( 
        locs_B, model.shift.d.numpy() , 0 )
        , dtype = tf.float32)
    
    model = Minimum_Entropy.Polynomial()  
    model_apply_grads = get_apply_grad_fn_nobatch()
    
    opt = tf.optimizers.Adam(learning_rate=0.001)
    loss = model_apply_grads(ch1, ch2_shift, model, opt, epochs)
    
    return model, loss
    


#%% optimization function
def get_apply_grad_fn_nobatch():
    #@tf.function
    def apply_grad(ch1, ch2, model, opt, epochs):
        '''
        The function that minimizes a certain model using TensorFlow GradientTape()
        This function does not use batches
        
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
        epochs : int
            Number of iterations for Gradient Descent

        Returns
        -------
        y : float32
            the relative entropy.

        '''
        print('Loading, this might take a while...')
        y1 =1 
        for i in range(epochs):
            with tf.GradientTape() as tape:
                y = model(ch1, ch2)
                        
            gradients = tape.gradient(y, model.trainable_variables)
                
            if i%10 == 0:
                print('i = ',i,' / ', epochs)
                print(y)
                print(model.trainable_variables)
          
            if np.abs(y1-y) < 4:
                break
            
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            y1 = y
        return y
    return apply_grad