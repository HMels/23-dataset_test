
# Minimum_Entropy.py
'''
This script is used to calculate the Mapping via the Minimum Entropy Method described by Cnossen2021

The script contains the next functions:
- KL_divergence() 
    the Kullback-Leibler divergence between localization i and j
- Rel_entropy()
    the relative entropy for certain localizations
    

The script also contains the next Model in the form of Classes
- Parameterized_module_simple:      the Parameterized model for calculating the minimum entropy
|- Shift
|- Rotation


- Polynomial_module:                the Polynomial model for calculating the minimum entropy
|- Polynomial
'''
import tensorflow as tf


#%% functions
@tf.autograph.experimental.do_not_convert
def Rel_entropy(ch1, ch2):
    return tf.reduce_sum(tf.square(ch1-ch2))


#%% Classes  
class Poly3Mod(tf.keras.Model):
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
        #ch2_mapped = self.transform_mat(ch2)
        ch2_mapped = self.transform_vec(ch2)
        return Rel_entropy(ch1, ch2_mapped)
    
    
    @tf.autograph.experimental.do_not_convert
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
    
    
    @tf.autograph.experimental.do_not_convert
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


#%% Polynomial 2
class Poly2Mod(tf.keras.Model):
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
        #ch2_mapped = self.transform_mat(ch2)
        ch2_mapped = self.transform_vec(ch2)
        return Rel_entropy(ch1, ch2_mapped)
    
    
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        y = tf.stack([
            tf.concat([self.M1[0,0]*tf.ones([x_input.shape[0],1]), 
                       self.M1[1,0]*x_input[:,0][:,None],
                       self.M1[0,1]*x_input[:,1][:,None],
                       self.M1[1,1]*(x_input[:,0]*x_input[:,1])[:,None]
                       ], axis = 1),
            tf.concat([self.M2[0,0]*tf.ones([x_input.shape[0],1]), 
                       self.M2[1,0]*x_input[:,0][:,None],
                       self.M2[0,1]*x_input[:,1][:,None],
                       self.M2[1,1]*(x_input[:,0]*x_input[:,1])[:,None]
                       ], axis = 1),
            ], axis = 2)
        return tf.reduce_sum(y, axis = 1)
    
    
    @tf.autograph.experimental.do_not_convert
    def transform_mat(self, x_input):
        y = tf.stack([
            tf.concat([self.M1[0,0]*tf.ones([1, x_input.shape[0], x_input.shape[1]]), 
                       self.M1[1,0]*x_input[:,:,0][None],
                       self.M1[0,1]*x_input[:,:,1][None],
                       self.M1[1,1]*(x_input[:,:,0]*x_input[:,:,1])[None]
                       ], axis = 0)[:,:,:,None],
            tf.concat([self.M2[0,0]*tf.ones([1, x_input.shape[0], x_input.shape[1]]), 
                       self.M2[1,0]*x_input[:,:,0][None],
                       self.M2[0,1]*x_input[:,:,1][None],
                       self.M2[1,1]*(x_input[:,:,0]*x_input[:,:,1])[None]
                       ], axis = 0)[:,:,:,None]
            ], axis = 3)
        return tf.reduce_sum(tf.reduce_sum(y, axis = 0), axis = 3)
    
    
#%% ShiftMod
class ShiftMod(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, name='shift'):
        super().__init__(name=name)
        
        self.d = tf.Variable([0,0], dtype=tf.float32, trainable=True, name='shift')


    @tf.function
    def call(self, ch1, ch2):
        #ch2_mapped = self.transform_mat( ch2 )
        ch2_mapped = self.transform_vec( ch2 )
        return Rel_entropy(ch1, ch2_mapped)
    
    
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        return x_input + self.d[None]
    
    
    @tf.autograph.experimental.do_not_convert
    def transform_mat(self, x_input):
        return x_input + self.d[None,None]
    
    
#%% RotationMod
class RotationMod(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, name='rotation'):
        super().__init__(name=name)
        
        self.theta = tf.Variable(0, dtype=tf.float32, trainable=True, name='rotation')
        
        
    @tf.function
    def call(self, ch1, ch2):
        #ch2_mapped = self.transform_mat( ch2 )
        ch2_mapped = self.transform_vec( ch2 )
        return Rel_entropy(ch1, ch2_mapped)
    
    
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        cos = tf.cos(self.theta * 0.0175)
        sin = tf.sin(self.theta * 0.0175)
        
        x1 = x_input[:,0]*cos - x_input[:,1]*sin
        x2 = x_input[:,0]*sin + x_input[:,1]*cos
        r = tf.stack([x1, x2], axis =1 )
        return r
    
    
    @tf.autograph.experimental.do_not_convert
    def transform_mat(self, x_input):
        cos = tf.cos(self.theta * 0.0175)
        sin = tf.sin(self.theta * 0.0175)
        
        x1 = x_input[:,:,0]*cos - x_input[:,:,1]*sin
        x2 = x_input[:,:,0]*sin + x_input[:,:,1]*cos
        r = tf.stack([x1, x2], axis =2 )
        return r
