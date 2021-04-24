# run_optimization.py
"""
Created on Wed Apr 14 15:40:25 2021

@author: Mels
"""
import tensorflow as tf
import Minimum_Entropy
#import Minimum_Entropy_eager as Minimum_Entropy
import numpy as np

#%% initialize function
def initialize_optimizer(locs_A, locs_B, Map_opt='Parameterized_simple' , Batch_on=False, batch_size=None,
                         num_batches=None, learning_rate=.001, epochs = 50):
    
    # decide what mapping to optimize
    if Map_opt == 'Parameterized_simple':  
        model = Minimum_Entropy.Parameterized_module_simple(name='Parameterized_simple')
    if Map_opt == 'Parameterized_complex':  
        model = Minimum_Entropy.Parameterized_module_complex(name='Parameterized_complex')
    if Map_opt == 'Polynomial':
        model = Minimum_Entropy.Polynomial_module(name='Polynomial')
    
    # decide if dataset will be split in batches
    if Batch_on:
        ch1 = split_grid(locs_A, num_batches, batch_size)
        ch2 = split_grid(locs_B, num_batches, batch_size)
        model_apply_grads = get_apply_grad_fn_batch()
    else:
        ch1 = tf.Variable( locs_A, dtype = tf.float32)
        ch2 = tf.Variable( locs_B, dtype = tf.float32)
        model_apply_grads = get_apply_grad_fn_nobatch()
    
    opt = tf.optimizers.Adam(learning_rate)
    loss = model_apply_grads(ch1, ch2, model, opt, epochs)
    
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


def get_apply_grad_fn_batch():
    @tf.function
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
        epochs = 200
        for i in range(epochs):
            for j in range(len(ch1)):
                with tf.GradientTape() as tape:
                    y = model(ch1[j], ch2[j])
                            
                gradients = tape.gradient(y, model.trainable_variables)
                     
                if i%10 == 0:
                    print('i = ',i,' / ', epochs)
              
                opt.apply_gradients(zip(gradients, model.trainable_variables))
                
        return y
    return apply_grad


#%% Split the system 
def split_grid(locs, num_batches, batch_size):
    '''
    Splits the channel into multiple regions

    Parameters
    ----------
    locs : Nx2 float numpy array
        Array containing the localizations.
    num_batches : 2 int numpy array
        Array containing the number of [x1,x2] regions we want the channel to be.
    batch_size : int 
        Maximum number of locs per region.

    Returns
    -------
    locs_grid : list
        A list containing multiple arrays, for which every array is the batch of a region.

    '''
    
    img = np.empty([2,2], dtype = float)        # calculate borders of system
    img[0,0] = np.min(locs[:,0])
    img[1,0] = np.max(locs[:,0])
    img[0,1] = np.min(locs[:,1])
    img[1,1] = np.max(locs[:,1])
    size_img = img[1,:] - img[0,:]
    
    grid_size = size_img/ num_batches
    grid_num = 0
    locs_grid = []
    for i in range(num_batches[0]):
        for j in range(num_batches[1]):
            
            l_grid = img[0,:] + np.array([ i*grid_size[0], j*grid_size[1] ])
            r_grid = img[0,:] + np.array([ (i+1)*grid_size[0], (j+1)*grid_size[1] ])
            
            indx1 = np.argwhere( (locs[:,0] > l_grid[0]) * (locs[:,1] > l_grid[1])
                                * (locs[:,0] <= r_grid[0]) * (locs[:,1] <= r_grid[1]) )
            new_locs = locs[ indx1 ]
            
            N = new_locs.shape[0]
            if batch_size <= N:     # limit the maximum locs per batch
                indx = np.random.choice(N, batch_size, replace=False)  
                new_locs = new_locs[indx]
                
            locs_grid.append( tf.Variable(new_locs[:,0,:], dtype = tf.float32) )
            grid_num += 1
    return locs_grid



    