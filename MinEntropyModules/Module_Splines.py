# SplinesMod.py

import tensorflow as tf

import MinEntropyModules.generate_neighbours as generate_neighbours
import MinEntropyModules.train_model as train_model
from MinEntropyModules.MinEntropy_fn import Rel_entropy


#%% run_optimization
def run_optimization(ch1, ch2, N_it=200, gridsize=50, maxDistance=50, learning_rate=1e-3,
                     direct=False, opt=tf.optimizers.Adagrad, sys_param=None):
    '''
    Parameters
    ----------
    ch1 ,ch2 : Nx2 Tensor
        Tensor containing the localizations of their channels.
    N_it : int, optional
        Number of iterations used in the training loop. The default 
    gridsize : float32, optional
        The size of the grid splines
    maxDistance : float32, optional
        The distance in which the Nearest Neighbours will be searched. 
        The default is 50nm.
    learning_rate : float, optional
        The initial learning rate of our optimizer. the default is 1.
    direct : bool, optional
        Do we want to run the algorithm with pairs or with a neighbours algorithm.
        The default is False.
    opt : tf.optimizers, optional
        The Optimizer used. The default is tf.optimizers.Adagrad
    sys_params : list, optional
        List containing the size of the system. The optional is None,
        which means it will be calculated by hand

    Returns
    -------
    mods : Models() Class
        Class containing the information and functions of the optimization models.
    ch2 : Nx2 float Tensor
        Mapped version of ch2

    '''  
    ch1_input = tf.Variable(ch1/gridsize, trainable=False)
    ch2_input = tf.Variable(ch2/gridsize, trainable=False)
    
    if sys_param is None:
        x1_min = tf.reduce_min(tf.floor(ch2_input[:,0]))
        x2_min = tf.reduce_min(tf.floor(ch2_input[:,1]))
        x1_max = tf.reduce_max(tf.floor(ch2_input[:,0]))
        x2_max = tf.reduce_max(tf.floor(ch2_input[:,1]))
    else:
        x1_min = tf.floor(sys_param[0,0]/gridsize)
        x2_min = tf.floor(sys_param[0,1]/gridsize)
        x1_max = tf.floor(sys_param[1,0]/gridsize)
        x2_max = tf.floor(sys_param[1,1]/gridsize)
        
    CP_idx = tf.cast(tf.stack(
        [( ch2_input[:,0]-x1_min+2)//1 , ( ch2_input[:,1]-x2_min+2)//1 ], 
        axis=1), dtype=tf.int32)
        
    if direct:          # direct 
        nn1=None
        nn2=None
        
        x1_grid = tf.range(x1_min-2, x1_max+4, 1)
        x2_grid = tf.range(x2_min-2, x2_max+4, 1)
        CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
        model = CatmullRomSplines_direct(CP_locs, CP_idx, ch2)
    
    else:               # Generate Neighbours
        neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
        ch1_input.numpy(), ch2_input.numpy(), maxDistance=maxDistance/gridsize, threshold=None, k=16)
        nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
        nn2 = tf.Variable( neighbours_B, dtype = tf.float32)
        
        
        x1_grid = tf.range(x1_min-2, x1_max+4, 1)
        x2_grid = tf.range(x2_min-2, x2_max+4, 1)
        CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
        
        CP_idx_nn = tf.cast(tf.stack(
            [( nn2[:,:,0]-x1_min+2)//1 , ( nn2[:,:,1]-x2_min+2)//1 ], 
            axis=2), dtype=tf.int32)  
        
        model = CatmullRomSplines(CP_locs, CP_idx, ch2, CP_idx_nn)

    
    # The Model
    mods = train_model.Models(model=model, learning_rate=learning_rate, opt=opt)
    
    # The Training Function
    model_apply_grads = train_model.get_apply_grad_fn()
    mods, ch2_input = model_apply_grads(ch1_input, ch2_input, N_it, mods, nn1, nn2)
    return mods, ch2_input*gridsize


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
        self.CP_locs_trainable = tf.Variable(CP_locs[2:-2,2:-2,:], dtype=tf.float32,
                                   trainable=True, name='ControlPointstrainable')  
        self.CP_locs_untrainable_ax0 = tf.Variable(
            [CP_locs[:2,:,:][None], CP_locs[-2:,:,:][None]],
            trainable=False, name='ControlPointsUntrainable_ax0'
            )
        self.CP_locs_untrainable_ax1 = tf.Variable(
            [CP_locs[2:-2,:2,:][:,None], CP_locs[2:-2,-2:,:][:,None]],
            trainable=False, name='ControlPointsUntrainable_ax1'
            )
        
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
    def call(self, nn1, nn2):
        nn2_mapped = self.transform_mat(nn2)
        return Rel_entropy(nn1, nn2_mapped)
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        self.load_CPlocs()
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
        self.load_CPlocs()
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
    @tf.autograph.experimental.do_not_convert
    def load_CPlocs(self):
        self.CP_locs = tf.concat([ 
            self.CP_locs_untrainable_ax0[0],
            tf.concat([ 
                self.CP_locs_untrainable_ax1[0], 
                self.CP_locs_trainable,
                self.CP_locs_untrainable_ax1[1]
                ],axis=1),
            self.CP_locs_untrainable_ax0[1]
            ],axis=0)
        
    
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
        
        
#%% CatmullRomSplines_direct
class CatmullRomSplines_direct(tf.keras.Model):
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
        self.CP_locs_trainable = tf.Variable(CP_locs[1:-1,1:-1,:], dtype=tf.float32,
                                   trainable=True, name='ControlPointstrainable')  
        self.CP_locs_untrainable_ax0 = tf.Variable(
            [CP_locs[0,:,:][None], CP_locs[-1,:,:][None]],
            trainable=False, name='ControlPointsUntrainable_ax0'
            )
        self.CP_locs_untrainable_ax1 = tf.Variable(
            [CP_locs[1:-1,0,:][:,None], CP_locs[1:-1,-1,:][:,None]],
            trainable=False, name='ControlPointsUntrainable_ax1'
            )
        
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
        ch2_mapped = self.transform_vec(ch2)
        return tf.reduce_sum(tf.square(ch1-ch2_mapped)) 
    
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        self.load_CPlocs()
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
        self.load_CPlocs()
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
    @tf.autograph.experimental.do_not_convert
    def load_CPlocs(self):
        self.CP_locs = tf.concat([ 
            self.CP_locs_untrainable_ax0[0],
            tf.concat([ 
                self.CP_locs_untrainable_ax1[0], 
                self.CP_locs_trainable,
                self.CP_locs_untrainable_ax1[1]
                ],axis=1),
            self.CP_locs_untrainable_ax0[1]
            ],axis=0)
        
    
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