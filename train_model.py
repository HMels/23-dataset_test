import tensorflow as tf

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
                if i%100==0: print('i = ',i)
                if i==4000:
                    print('Error: maximum iterations reached. Restart the training loop')
                    break
                
        elif nn1 is not None and nn2 is not None:
            while not mods.endloop:
                mods.Training_loop(nn1, nn2)                 # the training loop
                i+=1     
                if i%100==0: print('i = ',i)
                if i==1000:
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
        y1 = self.model(ch1, ch2)
        #self.Reject_fn(y1)
            
        
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
        self.opt_init.lr.assign(learning_rate)
        
        
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