import tensorflow as tf

#%% apply_grad function
def get_apply_grad_fn():
    #@tf.function
    def apply_grad(ch1, ch2, N_it, mods, nn1=None, nn2=None):
        '''

        Parameters
        ----------
        ch1 ,ch2 : Nx2 Tensor
            Tensor containing the localizations of their channels.
        mods : Model()
            The Model which will be optimized.
        N_it : int
            Number of iterations used.
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
        i=0
        if nn1 is None and nn2 is None:
            for i in range(N_it):
                mods.Training_loop(ch1, ch2)                 # the training loop
                i+=1
                if i==int(N_it/4): print('25%')
                if i==int(N_it/2): print('50%')
                if i==int(3*N_it/4): print('75%')
                if i==int(N_it): print('100%')
                
        elif nn1 is not None and nn2 is not None:
            for i in range(N_it):
                mods.Training_loop(nn1, nn2)                 # the training loop
                i+=1
                if i==int(N_it/4): print('25%')
                if i==int(N_it/2): print('50%')
                if i==int(3*N_it/4): print('75%')
                if i==int(N_it): print('100%')
                
                
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
        
        
    def Training_loop(self, ch1, ch2):
        with tf.GradientTape() as tape:
            self.entropy = self.model(ch1, ch2)
        
        self.grads = tape.gradient(self.entropy, self.var)
        self.opt_init.apply_gradients(zip(self.grads, self.var))
         
        
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