# pre_alignment.py
"""
Created on Tue May  4 17:43:36 2021

@author: Mels
"""
import tensorflow as tf

from run_optimization import Models

import Minimum_Entropy
import generate_neighbours
import run_optimization

#%% alignment functions 
def align_shift(ch1, ch2, mod = None, maxDistance=50):
    print('Aligning channels via Minimum Entropy Shift...')
    
    # Generate Neighbours 
    neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
    ch1.numpy(), ch2.numpy(), threshold=10, maxDistance=50)
    nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
    nn2 = tf.Variable( neighbours_B, dtype = tf.float32)
    
    if mod == None:
        model = Minimum_Entropy.ShiftMod('shift')
        opt = tf.optimizers.Adagrad
        learning_rate = 1.0
        mod = Models(model, learning_rate, opt)
        
        mod.Train(nn1, nn2)        
        ch2 = mod.model.transform_vec(ch2)
        print('Pre-Aligned!')
        print(mod.model.trainable_variables)
    
    elif mod.model.name == 'shift':
        mod.Train(nn1, nn2)        
        ch2 = mod.model.transform_vec(ch2)
        print('Pre-Aligned Shift!')        
        print(mod.model.trainable_variables)
        
    else: 
        print('Unknown model', mod.model.name,'used for shift alignment!')
    return ch2, mod


def align_rotation(ch1, ch2, mod = None, maxDistance=50):
    print('Aligning channels via Minimum Entropy Rotation...')
    
    # Generate Neighbours 
    neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
    ch1.numpy(), ch2.numpy(), threshold=None, maxDistance=50)
    nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
    nn2 = tf.Variable( neighbours_B, dtype = tf.float32)
        
    if mod == None:
        model = Minimum_Entropy.RotationMod('rotation')
        opt = tf.optimizers.Adagrad
        learning_rate = 1e-2
        mod = Models(model, learning_rate, opt)
        
        mod.Train(nn1, nn2)        
        ch2 = mod.model.transform_vec(ch2)
        print('Pre-Aligned Rotation!')
        print(mod.model.trainable_variables)
    
    elif mod.model.name == 'rotation':
        mod.Train(nn1, nn2)        
        ch2 = mod.model.transform_vec(ch2)
        print('Pre-Aligned!')        
        print(mod.model.trainable_variables)
        
    else: 
        print('Unknown model', mod.model.name,'used for alignment!')
    return ch2, mod
     
   
def align(ch1, ch2, mods = [], maxDistance=50):
    print('Aligning channels via Minimum Entropy Shift and Rotation...')
    # Generate Neighbours 
    neighbours_A, neighbours_B = generate_neighbours.find_bright_neighbours(
    ch1.numpy(), ch2.numpy(), maxDistance=maxDistance, threshold=None)
    nn1 = tf.Variable( neighbours_A, dtype = tf.float32)
    nn2 = tf.Variable( neighbours_B, dtype = tf.float32)
    
    if mods == None:
        mods=[]
        
        model = Minimum_Entropy.ShiftMod('shift')
        opt = tf.optimizers.Adagrad
        learning_rate = 1.0
        mods.append( Models(model, learning_rate, opt) )
        
        model = Minimum_Entropy.RotationMod('rotation')
        opt = tf.optimizers.Adagrad
        learning_rate = 1e-2
        mods.append( Models(model, learning_rate, opt) )
        
        ## Training Loop
        model_apply_grads = run_optimization.get_apply_grad_fn_dynamic()
        mods, ch2_map = model_apply_grads(ch1, ch2, nn1, nn2, mods) 
        
    else: 
         ## Training Loop
        model_apply_grads = run_optimization.get_apply_grad_fn_dynamic()
        ch2_map = model_apply_grads(ch1, ch2, nn1, nn2, mods) 
    return ch2_map, mods


def align1(ch1, ch2, mods = None, maxDistance=50):
    print('Aligning channels via Minimum Entropy Shift and Rotation...')
    
    if mods == None:
        mods=[]
        
        model = Minimum_Entropy.ShiftMod('shift')
        opt = tf.optimizers.Adagrad
        learning_rate = 1.0
        mods.append( Models(model, learning_rate, opt) )
        
        model = Minimum_Entropy.RotationMod('rotation')
        opt = tf.optimizers.Adagrad
        learning_rate = 1e-2
        mods.append( Models(model, learning_rate, opt) )
        
        ## Training Loop
        model_apply_grads = run_optimization.get_apply_grad_fn_dynamic()
        mods, ch2_map = model_apply_grads(ch1, ch2, mods) 
        
    else: 
         ## Training Loop
        model_apply_grads = run_optimization.get_apply_grad_fn_dynamic()
        mods, ch2_map = model_apply_grads(ch1, ch2, mods)
    return ch2_map, mods