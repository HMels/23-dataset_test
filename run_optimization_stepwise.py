# run_optimization_stepwise.py
"""
Created on Wed Apr 14 15:40:25 2021

@author: Mels
"""
import tensorflow as tf

import MinEntropy

#%% initialize function
def run_optimization(locs_A, locs_B,# neighbour_idx,
                     model, Batch_on=False, batch_size=None,
                         num_batches=None, learning_rate=.001, epochs = 50):
    
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)
    model_apply_grads = get_apply_grad_fn_dynamic()
   
    loss, var, ch2 = model_apply_grads(ch1, ch2)
    return model, loss, var, ch2


#%%
def get_apply_grad_fn_dynamic():
    #@tf.function
    def apply_grad(ch1, ch2):
        models = [MinEntropy.ShiftMod('shift'), 
                  MinEntropy.RotationMod('rotation'),
                  MinEntropy.Poly3Mod('polynomial')]
        learning_rates = [1.0, 1e-2, 5e-26]
        optimizers = [tf.optimizers.Adagrad,
                      tf.optimizers.Adagrad,
                      tf.optimizers.Adam]
        
        mods = []
        for i in range(len(models)):
            mods.append( Models(model=models[i], learning_rate = learning_rates[i], opt=optimizers[i] ))
            mods[i].var = mods[i].model.trainable_variables
           
        n = len(models)
        i=0
        endloop = False
        while not endloop:
            j = i%n                                         # for looping over the different models
            endloop = mods[0].endloop * mods[1].endloop * mods[1].endloop
            mods[j].Training_loop(ch1, ch2)                         # the training loop
            # is ch2 updated????
            i+=1     
            if i%150*n==0:
                print('i = ',i//n)                         
        print('completed in',i//n,' iterations')
        
        # delete this loop
        for i in range(len(models)):
            print('Model: ', mods[i].model)
            #print('+ entropy',mods[i].entropy)
            print('+ variables',mods[i].var)
            print('\n')
            ch2 = models[i].transform(ch2)
            
        return mods[-1], mods[-1].var, ch2
    return apply_grad


        
    
#%% 
class Models():
    def __init__(self, model=None, opt=None, learning_rate=None,
                 var=None, entropy=None, grads=None):
        self.model = model if model is not None else {}
        self.opt = opt if opt is not None else tf.optimizers.Adagrad    # uninitialized optimizer
        self.learning_rate = learning_rate if learning_rate is not None else 1e-3
        self.var = model.trainable_weights if model is not None else {}
        self.entropy = entropy if entropy is not None else {}
        self.grads = grads if grads is not None else {}
        self.opt_init = self.opt(self.learning_rate)                    # initialized optimizer
        self.Reject =  False
        self.endloop = False
        self.rejections = 0
        self.step_along_gradient = self.load_step_along_gradient(model)
        self.var_new = model.trainable_weights if model is not None else {}
        
    def Training_loop(self, ch1, ch2):        
        if not self.Reject:
            with tf.GradientTape() as tape:                         # compute new grads  
                self.entropy = self.model(ch1, ch2)    
            self.grads = tape.gradient(self.entropy, self.var)
            
        self.var_new = self.step_along_gradient(self.var, self.learning_rate,
                                                self.grads, self.var_new)
        self.opt_init.apply_gradients(zip(self.grads, self.var_new))
        
        y1 = self.model(ch1, ch2)                                   # compute new entropy
        self.Reject_fn(y1, self.var_new)                                 # Check if new entr is better
        
        
    def Reject_fn(self, y1, var_new):
        # Looks if the new entropy is better and adjusts the system accordingly
        if y1<self.entropy:
            self.Reject = False
            self.var = var_new                                      # accept new coefficients
            self.rejections = 0
            self.learning_rate = self.learning_rate*1.2
            self.opt_init = self.opt( self.learning_rate )
        else:
            self.Reject = True
            self.rejections+=1
            if self.rejections==20:                                 # convergence reached
                self.endloop = True                                 # accept current estimate
            self.learning_rate = self.learning_rate / 2
            self.opt_init = self.opt( self.learning_rate )
            
            
    def load_step_along_gradient(self, model):
        if model.name == 'rotation':
            def step_along_gradient(var, learning_rate, grads, var_new):      # step along gradient for rotation
                var_new[0].assign(var[0] + learning_rate*grads[0])
                return var_new
            return step_along_gradient
        elif model.name == 'shift':
            def step_along_gradient(var, learning_rate, grads, var_new):      # step along gradient for rotation
                var_new[0][0].assign(var[0][0] + learning_rate*grads[0][0])
                var_new[0][1].assign(var[0][1] + learning_rate*grads[0][1])
                return var_new
            return step_along_gradient
        elif model.name == 'polynomial':
            def step_along_gradient(var, learning_rate, grads, var_new):      # step along gradient for rotation
                var_new[0].assign(var[0] + learning_rate*grads[0])
                var_new[1].assign(var[1] + learning_rate*grads[1])
                return var_new
            return step_along_gradient
        else:
            print('Error: ',model.name,' module not in existence!')
        
