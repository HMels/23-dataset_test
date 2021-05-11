# run_optimization.py
"""
Created on Wed Apr 14 15:40:25 2021

@author: Mels
"""
import tensorflow as tf

import generate_neighbours


#%% functions1
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
    model_apply_grads = get_apply_grad_fn_dynamic()
    return model_apply_grads(ch1, ch2, nn1, nn2, mods) 


def get_apply_grad_fn_dynamic():
    #@tf.function
    def apply_grad(ch1, ch2, nn1, nn2, mods):
        print('Optimizing...')
        n=len(mods)
        i=0
        endloop = False
        while not endloop:
            j = i%n                                         # for looping over the different models
            if j==0: endloop=1
            endloop = endloop*mods[j].endloop
            mods[j].Training_loop(nn1, nn2)                         # the training loop
            
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
            self.reset_learning_rate(self.learning_rate*1.2)
            self.endloop = False
        else:
            self.Reject = True
            self.rejections+=1
            self.var = self.var_old.copy()
            self.reset_learning_rate(self.learning_rate/2)
            if self.rejections==100:                                 # convergence reached
                self.endloop = True
                
                
    def reset_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.opt_init = self.opt(self.learning_rate)
        
        
    def Train(self, nn1, nn2):
        while not self.endloop:
            self.Training_loop(nn1, nn2)