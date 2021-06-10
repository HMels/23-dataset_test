# run_optimization.py
"""
Created on Wed Apr 14 15:40:25 2021

@author: Mels
"""
import tensorflow as tf

import generate_neighbours
import MinEntropy_direct
import MinEntropy


#%% Splines
def run_optimization_Splines(ch1, ch2, gridsize=50, threshold=10, maxDistance=50, 
                             learning_rate=1e-3, direct=False):
    '''
    Parameters
    ----------
    ch1 ,ch2 : Nx2 Tensor
        Tensor containing the localizations of their channels.
    gridsize : float32, optional
        The size of the grid splines
    threshold : int, optional
        The amount of rejections before ending the optimization loop.
        The default is 10.
    maxDistance : float32, optional
        The distance in which the Nearest Neighbours will be searched. 
        The default is 50nm.
    learning_rate : float, optional
        The initial learning rate of our optimizer. the default is 1.
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
    ch1_input = tf.Variable(ch1/gridsize, trainable=False)
    ch2_input = tf.Variable(ch2/gridsize, trainable=False)

        
    CP_idx = tf.cast(tf.stack(
        [( ch2_input[:,0]-tf.reduce_min(tf.floor(ch2_input[:,0]))+1)//1 , 
         ( ch2_input[:,1]-tf.reduce_min(tf.floor(ch2_input[:,1]))+1)//1 ], 
        axis=1), dtype=tf.int32)
        
    if direct:          # Generate Neighbours 
        nn1=None
        nn2=None
        
        x1_grid = tf.range(tf.reduce_min(tf.floor(ch2_input[:,0])) -1,
                       tf.reduce_max(tf.floor(ch2_input[:,0])) +3, 1)
        x2_grid =  tf.range(tf.reduce_min(tf.floor(ch2_input[:,1]))-1,
                            tf.reduce_max(tf.floor(ch2_input[:,1])) +3, 1)
        CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
    
        model=MinEntropy_direct.CatmullRomSplines(CP_locs, CP_idx, ch2_input, CP_idx_nn=None)
    
    else:               # Direct
        neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
        ch1_input.numpy(), ch2_input.numpy(), maxDistance=maxDistance/gridsize, threshold=None)
        #nn1 = ch1_input[:,None]
        nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
        #nn2 = ch2_input[:,None]
        nn2 = tf.Variable( neighbours_B, dtype = tf.float32)
        
        x1_grid = tf.range(tf.reduce_min(tf.floor(ch2_input[:,0])) -1,
                           tf.reduce_max(tf.floor(ch2_input[:,0])) +3, 1)
        x2_grid = tf.range(tf.reduce_min(tf.floor(ch2_input[:,1])) -1,
                           tf.reduce_max(tf.floor(ch2_input[:,1])) +3, 1)
        CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
        
        CP_idx_nn = tf.cast(tf.stack(
            [( nn2[:,:,0]-tf.reduce_min(tf.floor(ch2_input[:,0]))+1)//1 , 
             ( nn2[:,:,1]-tf.reduce_min(tf.floor(ch2_input[:,1]))+1)//1 ], 
            axis=2), dtype=tf.int32)        
        model=MinEntropy.CatmullRomSplines(CP_locs, CP_idx, nn2, CP_idx_nn)
    
    
    # The Model
    mods = Models(model=model, learning_rate=learning_rate, 
                  opt=tf.optimizers.Adagrad, threshold=threshold)
    
    # The Training Function
    model_apply_grads = get_apply_grad_fn()
    mods, ch2_input = model_apply_grads(ch1, ch2_input, mods, nn1, nn2)
    return mods, ch2_input*gridsize


#%% ShiftRot
def run_optimization_ShiftRot(ch1, ch2, maxDistance=50, threshold=10, learning_rate=1, direct=False):
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
    
    if direct:          # Generate Neighbours
        nn1=None
        nn2=None
        
        model=MinEntropy_direct.ShiftRotMod()
        
    else:               # Direct
        neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
        ch1.numpy(), ch2.numpy(), maxDistance=maxDistance, threshold=None)
        #nn1 = ch1[:,None]
        nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
        #nn2 = ch2[:,None]
        nn2 = tf.Variable( neighbours_B, dtype = tf.float32)
        
        model=MinEntropy.ShiftRotMod()
    
    # The Model
    mods = Models(model=model, learning_rate=learning_rate, 
                  opt=tf.optimizers.Adagrad, threshold=threshold)
    
    ## Training Loop
    model_apply_grads = get_apply_grad_fn()
    return model_apply_grads(ch1, ch2, mods, nn1, nn2)


#%% apply_grad function
def get_apply_grad_fn():
    #@tf.function
    def apply_grad(ch1, ch2, mods, nn1=None, nn2=None):
        '''

        Parameters
        ----------
        ch1 ,ch2 : Nx2 Tensor
            Tensor containing the localizations of their channels.
        mods : Model()
            The Model which will be optimized.
        nn1 , nn2 : Nx2xM, optional
            Tensor containing the localizations of their channesl and
            their neighbours. The default is None.

        Returns
        -------
        mods : Models() Class
            Class containing the information and functions of the optimization models.
        ch2 : Nx2 float Tensor
            Mapped version of ch2

        '''
        print('Optimizing...')
        
        i=0
        if nn1 is None and nn2 is None:
            while not mods.endloop:
                mods.Training_loop(ch1, ch2)                 # the training loop
                i+=1     
                if i%1000==0: print('i = ',i)
                if i==25000:
                    print('Error: maximum iterations reached. Restart the training loop')
                    break
                
        elif nn1 is not None and nn2 is not None:
            while not mods.endloop:
                mods.Training_loop(nn1, nn2)                 # the training loop
                i+=1     
                if i%1000==0: print('i = ',i)
                if i==25000:
                    print('Error: maximum iterations reached. Restart the training loop')
                    break
                
                
        print('completed in',i,' iterations')
        ch2 = mods.model.transform_vec(ch2)
            
        return mods, ch2
    return apply_grad


#%% 
class Models():
    def __init__(self, model, learning_rate, opt,
                 var=None, entropy=None, grads=None, threshold=None):
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
        self.threshold = threshold if threshold is not None else 10
        
        
    def Training_loop(self, ch1, ch2):
        if not self.Reject:
            with tf.GradientTape() as tape:
                self.entropy = self.model(ch1, ch2)
            self.grads = tape.gradient(self.entropy, self.var)
            
        self.opt_init.apply_gradients(zip(self.grads, self.var))
        
        #print(self.var)
        y1 = self.model(ch1, ch2)
        self.Reject_fn(y1)
            
        
    def Reject_fn(self, y1):
        # Looks if the new entropy is better and adjusts the system accordingly
        if y1<self.entropy:
            self.Reject = False
            self.rejections = 0
            #self.reset_learning_rate(self.learning_rate)
            self.endloop = False
        else:
            self.Reject = True
            self.rejections+=1
            self.var = self.var_old.copy()
            #self.reset_learning_rate(self.learning_rate/2)
            if self.rejections==self.threshold:         # convergence reached
                self.endloop = True
                
                
    def reset_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.opt_init = self.opt(self.learning_rate)
        
        
    def Train(self, nn1, nn2):
        while not self.endloop:
            self.Training_loop(nn1, nn2)
            
            
    def reset_loop(self, learning_rate=None, threshold=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
            self.opt_init = self.opt(self.learning_rate)
        self.Reject =  False
        self.endloop = False
        self.rejections = 0
        self.threshold = threshold if threshold is not None else 10