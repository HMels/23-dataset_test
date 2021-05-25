# run_optimization.py
"""
Created on Wed Apr 14 15:40:25 2021

@author: Mels
"""
import tensorflow as tf
import numpy as np

import generate_neighbours
import MinEntropy_direct as MinEntropy


#%%
def initiate_model(models, learning_rates, optimizers):
    '''
    Initiate the MinEntropy Model consisting of an array of sub-models

    Parameters
    ----------
    models : tf.keras.Layers.layer List (can also be single element)
        A certain model described in this file.
    learning_rates : float numpy array (can also be single element)
        The learning rate per model.
    optimizers : tf.optimizers List (can also be single element)
        The optimizer to be used.

    Returns
    -------
    mods : List
        List containing the different initiated layers of the model.

    '''
    if not isinstance(models, list): models= [models]
    if isinstance(learning_rates, list): learning_rates= np.array(learning_rates)
    if not isinstance(learning_rates, np.ndarray): learning_rates= np.array([learning_rates])
    if not isinstance(optimizers, list): optimizers= [optimizers]
    
    mods = []
    for i in range(len(models)):
        mods.append( Models(model=models[i], learning_rate=learning_rates[i], 
                            opt=optimizers[i] ))
        mods[i].var = mods[i].model.trainable_variables
    return mods


#%% Splines
def run_optimization_splines(ch1, ch2, gridsize = 50):
    x1_grid = tf.range(tf.reduce_min(ch2[:,0])-1.5*gridsize,
                       tf.reduce_max(ch2[:,0])+2*gridsize, gridsize)
    x2_grid = tf.range(tf.reduce_min(ch2[:,1])-1.5*gridsize, 
                       tf.reduce_max(ch2[:,1])+2*gridsize, gridsize)
    CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
    CP_idx = tf.cast(tf.stack([( ch2[:,0]-tf.reduce_min(ch2[:,0]) )//gridsize+1, 
                               ( ch2[:,1]-tf.reduce_min(ch2[:,1]) )//gridsize+1], axis=1),
                     dtype=tf.int32) 
    
    mods = [Models(model=MinEntropy.CatmullRomSplines(CP_locs, CP_idx), 
                  learning_rate=1, opt=tf.optimizers.Adagrad)]
    
    model_apply_grads = get_apply_grad_fn_splines()
    
    return model_apply_grads(ch1, ch2, mods)
    
    
def get_apply_grad_fn_splines():
    #@tf.function
    def apply_grad(ch1, ch2, mods):
        print('Optimizing...')
        n=len(mods)
        i=0
        
        endloop = np.empty(n, dtype = bool)
        for d in range(n): endloop[d]=mods[d].endloop
        
        while not np.prod(endloop):
            j = i%n                                         # for looping over the different models
            
            mods[j].Training_loop(ch1, ch2)                 # the training loop
            endloop[j]=mods[j].endloop
            
            i+=1     
            if i%(50*n)==0: print('i = ',i//n)
                      
        print('completed in',i//n,' iterations')
            
        return mods, ch2
    return apply_grad


#%% functions
def run_optimization(ch1, ch2, mods, maxDistance = 50):
    '''
    Parameters
    ----------
    ch1 ,ch2 : Nx2 Tensor
        Tensor containing the localizations of their channels.
    mods : Models() Class
        Class containing the information and functions of the optimization models.

    Returns
    -------
    mods : Models() Class
        Class containing the information and functions of the optimization models.
    ch2 : Nx2 float Tensor
        Mapped version of ch2

    '''
    # Generate Neighbours 
    neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
    ch1.numpy(), ch2.numpy(), maxDistance=maxDistance, threshold=None)
    nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
    nn2 = tf.Variable( neighbours_B, dtype = tf.float32)
    
    ## Training Loop
    model_apply_grads = get_apply_grad_fn1()
    return model_apply_grads(ch1, ch2, nn1, nn2, mods) 


def get_apply_grad_fn():
    #@tf.function
    def apply_grad(ch1, ch2, nn1, nn2, mods):
        print('Optimizing...')
        n=len(mods)
        i=0
        
        endloop = np.empty(n, dtype = bool)
        for d in range(n): endloop[d]=mods[d].endloop
        
        while not np.prod(endloop):
            j = i%n                                         # for looping over the different models
            
            mods[j].Training_loop(nn1, nn2)                         # the training loop
            endloop[j]=mods[j].endloop
            
            i+=1     
            if i%(50*n)==0: print('i = ',i//n)
                      
        print('completed in',i//n,' iterations')
        
        # delete this loop
        for i in range(len(mods)):
            print('Model: ', mods[i].model)
            print('+ variables',mods[i].var)
            print('\n')
            ch2 = mods[i].model.transform_vec(ch2)
            
        return mods, ch2
    return apply_grad


def get_apply_grad_fn1():
    #@tf.function
    def apply_grad(ch1, ch2, nn1, nn2, mods):
        print('Optimizing...')
        n=len(mods)
        i=0
        
        endloop = np.empty(n, dtype = bool)
        for d in range(n): endloop[d]=mods[d].endloop
        
        while not np.prod(endloop):
            j = i%n                                         # for looping over the different models
            
            mods[j].Training_loop(ch1, ch2)                 # the training loop
            endloop[j]=mods[j].endloop
            
            i+=1     
            if i%(50*n)==0: print('i = ',i//n)
                      
        print('completed in',i//n,' iterations')
        
        # delete this loop
        for i in range(len(mods)):
            print('Model: ', mods[i].model)
            print('+ variables',mods[i].var)
            print('\n')
            ch2 = mods[i].model.transform_vec(ch2)
            
        return mods, ch2
    return apply_grad
    

#%% 
class Models():
    def __init__(self, model, learning_rate, opt,
                 var=None, entropy=None, grads=None):
        self.model = model 
        self.opt = opt 
        self.learning_rate = learning_rate
        self.opt_init = opt(self.learning_rate)
        self.var = model.trainable_weights
        self.entropy = entropy if entropy is not None else {}
        self.grads = grads if grads is not None else {}
        self.Reject =  False
        self.endloop = False
        self.rejections = 0
        self.var_old = model.trainable_weights.copy()
        
        
    def Training_loop(self, ch1, ch2):
        if not self.Reject:
            with tf.GradientTape() as tape:
                self.entropy = self.model(ch1, ch2)
            self.grads = tape.gradient(self.entropy, self.var)
            
        self.opt_init.apply_gradients(zip(self.grads, self.var))
        
        y1 = self.model(ch1, ch2)
        self.Reject_fn(y1)
            
        
    def Reject_fn(self, y1):
        # Looks if the new entropy is better and adjusts the system accordingly
        if y1<self.entropy:
            self.Reject = False
            self.rejections = 0
            #self.reset_learning_rate(self.learning_rate*1.05)
            self.endloop = False
        else:
            self.Reject = True
            self.rejections+=1
            self.var = self.var_old.copy()
            self.reset_learning_rate(self.learning_rate/2)
            if self.rejections==10:                                 # convergence reached
                self.endloop = True
                
                
    def reset_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.opt_init = self.opt(self.learning_rate)
        
        
    def Train(self, nn1, nn2):
        while not self.endloop:
            self.Training_loop(nn1, nn2)