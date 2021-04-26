# output_text.py
"""
Created on Thu Apr 22 14:26:22 2021

@author: Mels
"""
import numpy as np
import tensorflow as tf
import time

import dataset_manipulation

#%% error handling of batches
def Info_batch(N, num_batches, batch_size, Batch_on=True):
    '''
    Handles the error if batches are sufficient large

    Parameters
    ----------
    N : int
        total amount of points.
    num_batches : 2 int array
        containing the amount of [x1,x2] batches.
    batch_size : int
        max number of points per batch.
    Batch_on : bool, optional
        Are batches used. The default is True.

    Returns
    -------
    None.

    '''
    if Batch_on:
        perc = np.prod(num_batches)*batch_size / N * 100
        if perc < 100:
            print('I: in the current setup (',num_batches[0],'x',num_batches[1],'batches, ',
                  batch_size,' points per batch and ',N, ' points total)',
                  ' an estimate of ', round(perc,2),'\% of points are used to calculate the minimum entropy.\n')
            input("Press Enter to continue...")
        else: 
            print('I: in the current setup (',num_batches[0],'x',num_batches[1],'batches, ',
                  batch_size,' points per batch and ',N, ' points total)',
                  ' an estimate of ', round(perc,2),'\% of points are used to calculate the minimum entropy. \nThe setup seems to be OK and the amount of Batches is sufficient.\n')
            time.sleep(2)
    else: 
        print('I: the total system contains', N, ' points. The setup seems to be OK',
              '\nNote that for big N, batches shoud be used.\n')
        time.sleep(2)


#%%
def generate_output(locs_A, locs_B, model, shift=np.array([0,0]), 
                    angle=0, shear=np.array([0,0]), scaling=np.array([1,1]),
                    print_output = True
                    ):
    '''
    generates output channel and text output

    Parameters
    ----------
    locs_A ,locs_B : Nx2 float np.array
        array containing the localizations in channel A and B.
    model : tf.keras.layer.Layer
        The model used for optimizing a mapping.
    shift : 2 float array, optional
        shift of image in nm. The default is np.array([0,0]).
    angle : float, optional
        angle of rotation in radians. The default is 0.
    shear : 2 float array, optional
        amount of shear. The default is np.array([0,0]).
    scaling : 2 float array, optional
        amount of scaling. The default is np.array([1,1]).
    print_output : bool
        True if you want to print the results vs comparisson

    Returns
    -------
    ch1 : ch2 : Nx2 TensorFlow Tensor
        array containing the localizations in channel A and B.
    ch2_mapped_model : Nx2 TensorFlow Tensor
        array containing the localizations of the mapped channel B.

    '''
    
    ch1 = tf.Variable( locs_A, dtype = tf.float32)
    ch2 = tf.Variable( locs_B, dtype = tf.float32)    

    ch2_mapped_model = tf.Variable( dataset_manipulation.simple_translation( 
        locs_B, model.shift.d.numpy() , model.rotation.theta.numpy() )
        , dtype = tf.float32)
        
    #ch2_mapped_model = tf.Variable( dataset_manipulation.polynomial_translation(
      #  locs_B, model.polynomial.M1, model.polynomial.M2) )
        
    return ch1, ch2, ch2_mapped_model

