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
def generate_output(localizations_A, localizations_B, model,
                    Map_opt='Parameterized_simple', shift=np.array([0,0]), 
                    angle=0, shear=np.array([0,0]), scaling=np.array([1,1]),
                    print_output = True
                    ):
    '''
    generates output channel and text output

    Parameters
    ----------
    localizations_A ,localizations_B : Nx2 float np.array
        array containing the localizations in channel A and B.
    model : tf.keras.layer.Layer
        The model used for optimizing a mapping.
    Map_opt : str , optional
        Which mapping is used. The default is 'Parameterized_simple'.
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
    
    ch1 = tf.Variable( localizations_A, dtype = tf.float32)
    ch2 = tf.Variable( localizations_B, dtype = tf.float32)
            
    ch2_mapped = tf.Variable(
        dataset_manipulation.complex_translation(localizations_B, 
                                                 -1 * shift, -1 * angle, 
                                                 -1 * shear, 1 / scaling) , 
        dtype = tf.float32)
    
    
    if Map_opt == 'Parameterized_simple':
        ch2_mapped_model = tf.Variable( dataset_manipulation.simple_translation( 
            localizations_B, model.shift.d.numpy() , model.rotation.theta.numpy() )
            , dtype = tf.float32)
       
        if print_output:
            print_result_parameterized_simple(model, shift, angle, ch1, 
                                              ch2_mapped_model, ch2_mapped)
        
    if Map_opt == 'Parameterized_complex':
        ch2_mapped_model = tf.Variable( dataset_manipulation.complex_translation(
            localizations_B, model.shift.d, model.rotation.theta, model.shear.shear,
            model.scaling.scaling) , dtype = tf.float32)
        if print_output:
            print_result_parameterized_complex(model, shift, angle, shear, scaling, 
                                               ch1, ch2_mapped_model, ch2_mapped)
    
    if Map_opt == 'Polynomial':
        ch2_mapped_model = tf.Variable( dataset_manipulation.polynomial_translation(
            localizations_B, model.polynomial.M1, model.polynomial.M2) )
        
    return ch1, ch2, ch2_mapped_model #, ch2_mapped

#%% output functions 
def print_result_parameterized_simple(model, shift, angle, ch1, ch2_mapped_model, 
                                      ch2_mapped, pix_size = 100):
    print('\n-------------------- RESULT --------------------------')
    print('+ Shift = ',-1 * model.shift.d.numpy() * pix_size, ' [nm]')
    print('+ Rotation = ',-1 * model.rotation.theta.numpy()*180/np.pi,' [degrees]')
    
    print('\n-------------------- COMPARISSON ---------------------')
    print('+ Shift = ', shift * pix_size, ' [nm]')
    print('+ Rotation = ', angle*180/np.pi, ' [degrees]')
    
    diff_d = (np.abs(model.shift.d.numpy() + shift) * pix_size)**2
    error_d = np.sqrt(diff_d[0] + diff_d[1])
    error_theta = np.abs(model.rotation.theta.numpy() + angle)*180/np.pi
    
    print('The error equals ', error_d,' nm and ', error_theta, ' degrees')
    
    
def print_result_parameterized_complex(model, shift, angle, shear, scaling, 
                                       ch1, ch2_mapped_model, ch2_mapped, pix_size = 100):
    print('\n-------------------- RESULT --------------------------')
    print('+ Shift = ',-1 * model.shift.d.numpy() * pix_size, ' [nm]')
    print('+ Rotation = ',-1 * model.rotation.theta.numpy()*180/np.pi,' [degrees]')
    print('+ Shear = ', -1 * model.shear.shear.numpy())
    print('+ Scaling = ', 1 / model.scaling.scaling.numpy())
    
    print('\n-------------------- COMPARISSON ---------------------')
    print('+ Shift = ', shift * pix_size, ' [nm]')
    print('+ Rotation = ', angle*180/np.pi, ' [degrees]')
    print('+ Shear = ', shear)
    print('+ Scaling = ', scaling)