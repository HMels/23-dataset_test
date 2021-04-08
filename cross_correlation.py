# cross_correlation.py
"""
This Module calculates the cross-correlation between two channels. 
The output is meant to only calculate the shift

The functions in this module are
- cross_corr_script()
- cross_corr()
- shift_output()
- output_image()
"""

import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt

import functions


def cross_corr_script(channel_A, channel_B, img_param, shift, pix_search = 1.5, output_on = True):
    '''
    the script for calculating the cross correlation between two channels 
    and gives the figure and text output 

    Parameters
    ----------
    channel_A, channel_B : NxN Int/Float Matrix
        Matrices containing the localizations of Channel A and B.
    img_param : Image Class 
        Contains all parameters of our system/image.
    shift : 2 Float Vector
        Contains the amount of shift between the channels.
    pix_search : Float, optional
        The amount of pixels we limit our search of shift in. The default is 1.5.
    output_on : Bool, optional
        True gives an output image and string, False supresses this. The default is True.

    Returns
    -------
    shift_calc : Float
        The shift calculated via Cross-Correlation.
    abs_error_nm : Float
        The absolute error in nm between the real shift and the calculated shift.
        
    '''    
    pix_search = pix_search * img_param.pix_size / img_param.zoom 
    bounds = functions.generate_bounds(img_param, pix_search = 1.5)

    # calculate cross correlation
    xcorr = cross_corr(channel_A, channel_B)
    
    # calculate the maximum cross-correlation and shift
    xcorr_window, max_loc, shift_calc, abs_error_nm = shift_output(bounds, xcorr, shift, img_param, output_on)
    
    # generate it in an image
    if output_on == True:
        output_image(channel_A, channel_B, xcorr_window, max_loc)
    
    return shift_calc, abs_error_nm


def cross_corr(channel_A, channel_B):
    '''
    Calculates the cross-correlation between two images via fft

    Parameters
    ----------
    channel_A, channel_B : NxN Int/Float Matrix
        Matrices containing the localizations of Channel A and B.

    Returns
    -------
    NxN Float Matrix
        Matrix containing the Cross-Correlation between the Channels.

    '''
    FimageA = fft.fft2(channel_A) 
    CFimageB = np.conj(fft.fft2(channel_B))
    return fft.fftshift( np.real(fft.ifft2((FimageA * CFimageB)))) / np.sqrt(channel_A.size) 


def shift_output(bounds, xcorr, shift, img_param, output_on = True): 
    '''
    calculates the shift of the two channels,
    It prints a string with the true shift vs calculated shift and the error
    
    Parameters
    ----------
    bounds : 2x2 Int Matrix
        Matrix containing the bounds in which we search for the optimal Cross-Correlation.
    xcorr : NxN Float Matrix
        Matrix containing the Cross-Correlation between the Channels.
    shift : 2 Float Vector
        Contains the amount of shift between the channels.
    img_param : Image Class 
        Contains all parameters of our system/image.
    output_on : Bool, optional
        True gives an output string, False supresses this. The default is True.

    Returns
    -------
    xcorr_window : Float Matrix
        the windowed part of the cross correlation.
    max_loc : int 2-Vector
         The indices of the maximum correlation location.
    shift_calc : Float
        The shift calculated via Cross-Correlation.
    abs_error_nm : Float
        The absolute error in nm between the real shift and the calculated shift.

    '''
    xcorr_window = xcorr[bounds[0,0]:bounds[0,1], bounds[1,0]:bounds[1,1]]
    
    max_loc = np.unravel_index(np.argmax(xcorr_window), xcorr_window.shape) 
    shift_calc = - np.array([max_loc[0] - xcorr_window.shape[0] / 2 ,
                             max_loc[1] - xcorr_window.shape[1] / 2]) # in units [zoom]
    abs_error_nm = round( np.sqrt((shift_calc - shift)[0]**2 + (shift_calc - shift)[1]**2) * img_param.zoom )
    
    if output_on == True:
        print('Cross-Correlation:')
        print('  the shift equals ',shift_calc / img_param.pix_size_zoom(), 'pixels, or ', shift_calc * img_param.zoom, 'nm')
        print('  the shift should be ',shift / img_param.pix_size_zoom(), 'pixels, or ', shift * img_param.zoom, 'nm')
        print('  the absolute error equals ', abs_error_nm, 'nm')
    
    return xcorr_window, max_loc, shift_calc, abs_error_nm


def output_image(channel_A, channel_B, xcorr_window, max_loc):
    '''
    generates three plots containing channel A, channel B, 
    and the cross correlation in a window

    Parameters
    ----------
    channel_A, channel_B : NxN Int/Float Matrix
        Matrices containing the localizations of Channel A and B.
    xcorr_window : Float Matrix
        the windowed part of the cross correlation.
    max_loc : int 2-Vector
         The indices of the maximum correlation location.

    Returns
    -------
    None.

    '''
    plt.figure()
    plt.subplot(131)
    plt.imshow(channel_A)
    plt.subplot(132)
    plt.imshow(channel_B)
    plt.subplot(133)
    plt.imshow(xcorr_window)
    plt.plot(max_loc[1], max_loc[0], marker='x', markersize=3, color="red")