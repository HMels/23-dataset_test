
# Minimum_Entropy.py
'''
This script is used to calculate the Mapping via the Minimum Entropy Method described by Cnossen2021

The script contains the next functions:
- KL_divergence() 
    the Kullback-Leibler divergence between localization i and j
- Rel_entropy()
    the relative entropy for certain localizations
    

The script also contains the next Model in the form of Classes
- ShiftRot:     The Shift and Rotation model for calculating the minimum entropy
- Splines:      The Catmull Rom Splines model 
- Pol3Mod:      The Polynomial model for calculating the minimum entropy
'''

import tensorflow as tf

#%% functions
@tf.function
def Rel_entropy(ch1, ch2):
    return tf.reduce_sum(tf.square(ch1-ch2))


#%% CatmullRomSplines
class CatmullRomSplines(tf.keras.Model):
    '''
    Main Layer for calculating the relative entropy of a certain deformation
    ----------
    - it takes the x_input, the [x1,x2] locations of all localizations
    - gives it a shift and rotation deformation
    - calculates the relative entropy via Rel_entropy()    
    '''
    def __init__(self, CP_locs, CP_idx, ch2, CP_idx_nn=None, name='CatmullRomSplines'):
        super().__init__(name=name)

        # The location of the ControlPoints. This will be trained
        self.CP_locs = tf.Variable(CP_locs, dtype=tf.float32,
                                   trainable=True, name='ControlPoints')  
        # The indices of which locs in ch2 belong to which CP_locs
        self.CP_idx = tf.Variable(CP_idx, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
        self.CP_idx_nn = (tf.Variable(CP_idx_nn, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
                          if CP_idx_nn is not None else {})
        self.A = tf.Variable([
            [-.5, 1.5, -1.5, 0.5],
            [1, -2.5, 2, -.5],
            [-.5, 0, .5, 0],
            [0, 1, 0, 0]
            ], trainable=False, dtype=tf.float32)
        self.r = tf.Variable(ch2%1, trainable=False, dtype=tf.float32, 
                             name='Distance to ControlPoinst')

    @tf.function
    def call(self, ch1, ch2):
        #ch2_mapped = self.transform_mat( self.r )
        ch2_mapped = self.transform_vec( self.r )
        return Rel_entropy(ch1, ch2_mapped)
    
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        self.update_splines(self.CP_idx)        
        x = x_input[:,0][:,None]%1
        y = x_input[:,1][:,None]%1
        
        M_matrix = tf.stack([
            tf.pow(x,3)*tf.pow(y,3)*self.Sum_A(0,0),
            tf.pow(x,3)*tf.pow(y,2)*self.Sum_A(0,1),
            tf.pow(x,3)*tf.pow(y,1)*self.Sum_A(0,2),
            tf.pow(x,3)*tf.pow(y,0)*self.Sum_A(0,3),
            
            tf.pow(x,2)*tf.pow(y,3)*self.Sum_A(1,0),
            tf.pow(x,2)*tf.pow(y,2)*self.Sum_A(1,1),
            tf.pow(x,2)*tf.pow(y,1)*self.Sum_A(1,2),
            tf.pow(x,2)*tf.pow(y,0)*self.Sum_A(1,3),
        
            tf.pow(x,1)*tf.pow(y,3)*self.Sum_A(2,0),
            tf.pow(x,1)*tf.pow(y,2)*self.Sum_A(2,1),
            tf.pow(x,1)*tf.pow(y,1)*self.Sum_A(2,2),
            tf.pow(x,1)*tf.pow(y,0)*self.Sum_A(2,3),
            
            tf.pow(x,0)*tf.pow(y,3)*self.Sum_A(3,0),
            tf.pow(x,0)*tf.pow(y,2)*self.Sum_A(3,1),
            tf.pow(x,0)*tf.pow(y,1)*self.Sum_A(3,2),
            tf.pow(x,0)*tf.pow(y,0)*self.Sum_A(3,3),
            ], axis=2)
        return tf.reduce_sum(M_matrix, axis=2)
    
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def transform_mat(self, x_input):
        self.update_splines(self.CP_idx_nn)        
        x = x_input[:,:,0][:,:,None]%1
        y = x_input[:,:,1][:,:,None]%1
        
        M_matrix = tf.stack([
            tf.pow(x,3)*tf.pow(y,3)*self.Sum_A(0,0),
            tf.pow(x,3)*tf.pow(y,2)*self.Sum_A(0,1),
            tf.pow(x,3)*tf.pow(y,1)*self.Sum_A(0,2),
            tf.pow(x,3)*tf.pow(y,0)*self.Sum_A(0,3),
            
            tf.pow(x,2)*tf.pow(y,3)*self.Sum_A(1,0),
            tf.pow(x,2)*tf.pow(y,2)*self.Sum_A(1,1),
            tf.pow(x,2)*tf.pow(y,1)*self.Sum_A(1,2),
            tf.pow(x,2)*tf.pow(y,0)*self.Sum_A(1,3),
        
            tf.pow(x,1)*tf.pow(y,3)*self.Sum_A(2,0),
            tf.pow(x,1)*tf.pow(y,2)*self.Sum_A(2,1),
            tf.pow(x,1)*tf.pow(y,1)*self.Sum_A(2,2),
            tf.pow(x,1)*tf.pow(y,0)*self.Sum_A(2,3),
            
            tf.pow(x,0)*tf.pow(y,3)*self.Sum_A(3,0),
            tf.pow(x,0)*tf.pow(y,2)*self.Sum_A(3,1),
            tf.pow(x,0)*tf.pow(y,1)*self.Sum_A(3,2),
            tf.pow(x,0)*tf.pow(y,0)*self.Sum_A(3,3),
            ], axis=3)
        return tf.reduce_sum(M_matrix, axis=3)
        
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def Sum_A(self,a,b):
        A_matrix = tf.stack([
            self.A[a,0]*self.A[b,0]*self.q00,
            self.A[a,0]*self.A[b,1]*self.q01,
            self.A[a,0]*self.A[b,2]*self.q02,
            self.A[a,0]*self.A[b,3]*self.q03,
            
            self.A[a,1]*self.A[b,0]*self.q10,
            self.A[a,1]*self.A[b,1]*self.q11,
            self.A[a,1]*self.A[b,2]*self.q12,
            self.A[a,1]*self.A[b,3]*self.q13,
            
            self.A[a,2]*self.A[b,0]*self.q20,
            self.A[a,2]*self.A[b,1]*self.q21,
            self.A[a,2]*self.A[b,2]*self.q22,
            self.A[a,2]*self.A[b,3]*self.q23,
            
            self.A[a,3]*self.A[b,0]*self.q30,
            self.A[a,3]*self.A[b,1]*self.q31,
            self.A[a,3]*self.A[b,2]*self.q32,
            self.A[a,3]*self.A[b,3]*self.q33
            ], axis=2)
        return tf.reduce_sum(A_matrix, axis=2)
    
    
    #@tf.function
    def update_splines(self, idx):
        self.q00 = tf.gather_nd(self.CP_locs, idx+[-1,-1])  # q_k
        self.q01 = tf.gather_nd(self.CP_locs, idx+[-1,0])  # q_k
        self.q02 = tf.gather_nd(self.CP_locs, idx+[-1,1])  # q_k
        self.q03 = tf.gather_nd(self.CP_locs, idx+[-1,2])  # q_k
            
        self.q10 = tf.gather_nd(self.CP_locs, idx+[0,-1])  # q_k
        self.q11 = tf.gather_nd(self.CP_locs, idx+[0,0])  # q_k
        self.q12 = tf.gather_nd(self.CP_locs, idx+[0,1])  # q_k
        self.q13 = tf.gather_nd(self.CP_locs, idx+[0,2])  # q_k
            
        self.q20 = tf.gather_nd(self.CP_locs, idx+[1,-1])  # q_k
        self.q21 = tf.gather_nd(self.CP_locs, idx+[1,0])  # q_k
        self.q22 = tf.gather_nd(self.CP_locs, idx+[1,1])  # q_k
        self.q23 = tf.gather_nd(self.CP_locs, idx+[1,2])  # q_k
            
        self.q30 = tf.gather_nd(self.CP_locs, idx+[2,-1])  # q_k
        self.q31 = tf.gather_nd(self.CP_locs, idx+[2,0])  # q_k
        self.q32 = tf.gather_nd(self.CP_locs, idx+[2,1])  # q_k
        self.q33 = tf.gather_nd(self.CP_locs, idx+[2,2])  # q_k
        
        
    def reset_CP(self, CP_idx, CP_idx_nn=None):
        # The indices of which locs in ch2 belong to which CP_locs
        self.CP_idx = tf.Variable(CP_idx, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
        self.CP_idx_nn = (tf.Variable(CP_idx_nn, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
                          if CP_idx_nn is not None else {})


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
        ch2_mapped =self.transform_vec(ch2)
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
    
    
#%% Poly3Mod  
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
    
    
    @tf.function
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
    
    
    @tf.function
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