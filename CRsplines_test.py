# catmull_rom_splines

import tensorflow as tf
import matplotlib.pyplot as plt

import generate_data
from setup_image import Deform


#%% Import dataset
path = [ 'C:/Users/Mels/Documents/example_MEP/mol115_combined_clusters.hdf5' ]

locs_A, locs_B = generate_data.run_channel_generation(
    path, deform=Deform(), error=0, Noise=0, realdata=True, subset=1, pix_size=1
    )

ch1 = tf.Variable( locs_A, dtype = tf.float32)
ch2 = tf.Variable( locs_B, dtype = tf.float32)


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
    def __init__(self, CP_locs, CP_idx, ch2, name='CatmullRomSplines'):
        super().__init__(name=name)

        # The location of the ControlPoints. This will be trained
        self.CP_locs = tf.Variable(CP_locs, dtype=tf.float32,
                                   trainable=True, name='ControlPoints')  
        # The indices of which locs in ch2 belong to which CP_locs
        self.CP_idx = tf.Variable(CP_idx, dtype=tf.int32,
                                  trainable=False, name='ControlPointsIdx')
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
        ch2_mapped = self.transform_vec( ch2 )
        return Rel_entropy(ch1, ch2_mapped)
    
    
    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def transform_vec(self, x_input):
        self.update_splines()        
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
    def update_splines(self):
        self.q00 = tf.gather_nd(self.CP_locs, self.CP_idx+[-1,-1])  # q_k
        self.q01 = tf.gather_nd(self.CP_locs, self.CP_idx+[-1,0])  # q_k
        self.q02 = tf.gather_nd(self.CP_locs, self.CP_idx+[-1,1])  # q_k
        self.q03 = tf.gather_nd(self.CP_locs, self.CP_idx+[-1,2])  # q_k
            
        self.q10 = tf.gather_nd(self.CP_locs, self.CP_idx+[0,-1])  # q_k
        self.q11 = tf.gather_nd(self.CP_locs, self.CP_idx+[0,0])  # q_k
        self.q12 = tf.gather_nd(self.CP_locs, self.CP_idx+[0,1])  # q_k
        self.q13 = tf.gather_nd(self.CP_locs, self.CP_idx+[0,2])  # q_k
            
        self.q20 = tf.gather_nd(self.CP_locs, self.CP_idx+[1,-1])  # q_k
        self.q21 = tf.gather_nd(self.CP_locs, self.CP_idx+[1,0])  # q_k
        self.q22 = tf.gather_nd(self.CP_locs, self.CP_idx+[1,1])  # q_k
        self.q23 = tf.gather_nd(self.CP_locs, self.CP_idx+[1,2])  # q_k
            
        self.q30 = tf.gather_nd(self.CP_locs, self.CP_idx+[2,-1])  # q_k
        self.q31 = tf.gather_nd(self.CP_locs, self.CP_idx+[2,0])  # q_k
        self.q32 = tf.gather_nd(self.CP_locs, self.CP_idx+[2,1])  # q_k
        self.q33 = tf.gather_nd(self.CP_locs, self.CP_idx+[2,2])  # q_k


#%% CatmullRomSplines
'''
ch1copy = tf.stack([ch1[:,0], ch1[:,1]], axis=1)
gridsize = 50               # nm, size between controlpoints
x_input = tf.Variable(ch2 / gridsize)

'''
gridsize = 1
x_input_original = tf.Variable([[.5, 1.5], [1.5, 0.5], [1.5, 1.5], [0.5, 0.5],
                       [3.5,4.5], [0.5,4.5], [0.5,3.5], [3.5,1.5]], dtype=tf.float32)
x_input = tf.Variable(x_input_original)
ch1copy = tf.Variable(tf.stack([x_input_original[:,0]+.02, x_input_original[:,1]-.04], axis=1))

#'''
x1_grid = tf.range(tf.reduce_min(tf.floor(x_input[:,0])) -1,
                   tf.reduce_max(tf.floor(x_input[:,0])) +3, 1)
x2_grid =  tf.range(tf.reduce_min(tf.floor(x_input[:,1]))-1,
                   tf.reduce_max(tf.floor(x_input[:,1])) +3, 1)
CP_locs = tf.transpose(tf.stack(tf.meshgrid(x1_grid,x2_grid), axis=2), [1,0,2])
CP_idx = tf.cast(tf.stack(
    [( x_input[:,0]-tf.reduce_min(tf.floor(x_input[:,0]))+1)//1 , 
     ( x_input[:,1]-tf.reduce_min(tf.floor(x_input[:,1]))+1)//1 ], 
    axis=1), dtype=tf.int32)


#%% The Optimization loop  
def Reject_fn(y1, entropy, Reject, rejections, endloop, 
              var, var_old, learning_rate, opt_init, opt):
    # Looks if the new entropy is better and adjusts the system accordingly
    if y1<entropy:
        Reject = False
        rejections = 0
        #self.reset_learning_rate(self.learning_rate*1.05)
        endloop = False
    else:
        Reject = True
        rejections+=1
        var = var_old.copy()
        
        learning_rate = learning_rate/2
        opt_init = opt(learning_rate)
    
        if rejections==1:                                 # convergence reached
            endloop = True
    return Reject, rejections, endloop, var, learning_rate, opt_init
  

model=CatmullRomSplines(CP_locs, CP_idx, x_input)
opt=tf.optimizers.Adagrad
learning_rate=1e-2
opt_init=opt(learning_rate)     
     
i=0
endloop=False
Reject=False
rejections=0
while not endloop:
    if not Reject:
        with tf.GradientTape() as tape:
            entropy = model(ch1copy, x_input)
        grads = tape.gradient(entropy, model.trainable_variables)
        
    opt_init.apply_gradients(zip(grads, model.trainable_variables))
    
    var_old = model.trainable_variables.copy()
    y1 = model(ch1copy, x_input)
    Reject, rejections, endloop, var, learning_rate, opt_init= (
        Reject_fn(y1, entropy, Reject, rejections, endloop,
                  model.trainable_variables, var_old, learning_rate, opt_init, opt) 
        )
            # the training loop
    i+=1     
    print('i = ',i)
    

    
print('completed in',i,' iterations')
x_mapped = model.transform_vec(x_input)
        

#%% Output      
plt.close('all')

plt.figure()
plt.plot(ch1copy[:,1]-x_mapped[:,1]*gridsize, ch1copy[:,0]-x_mapped[:,0]*gridsize, 'x')

plt.figure()
plt.plot(x_mapped[:,1]*gridsize,x_mapped[:,0]*gridsize, 'r.', label='Mapped')
plt.plot(ch1copy[:,1],ch1copy[:,0], 'b.', label='Original')

W = model.trainable_variables[0]
plt.plot(W[0,:,1]*gridsize, W[0,:,0]*gridsize, 'y+', label='Grid Mapped')
for i in range(1,CP_locs.shape[0]):
    plt.plot(W[i,:,1]*gridsize, W[i,:,0]*gridsize, 'y+')
 

plt.plot(CP_locs[0,:,1]*gridsize, CP_locs[0,:,0]*gridsize, 'g+', label='Grid Original')
for i in range(1,CP_locs.shape[0]):
    plt.plot(CP_locs[i,:,1]*gridsize, CP_locs[i,:,0]*gridsize, 'g+')
plt.legend()