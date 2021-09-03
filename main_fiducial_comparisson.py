# -*- coding: utf-8 -*-
"""
Created on Wed 18/08/2021

@author: Mels

This program tests our MinEntropy Model on the dataset from the paper 'Nanometer-accuracy 
distance measurements between fluorophores at the single-molecule level' and compares 
it to the results gotten in the same paper.

The paper takes a first dataset containing fiducial markers and aligns the channel (called registration)
roughly at first using a global affine transform. It then coupled all points via a simple 
kNN algorithm with k=1, after which it uses multiple local affine transforms to do the final alignments.
The resulting mapping is then implemented on a second fiducial marker dataset that is taken after 
the first. This resulting dataset is used to get a registration error from.

Our model will implement some kind of the same method. First we will roughly align the 
datasets using rcc. Then we will couple points via a kNN algorithm with k=1. The big 
difference now will be that we will locally align using a spline grid. The resulting 
mapping will also be tested on the second dataset. 

"""
import sys
sys.path.append('../')

# Packages
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import pandas as pd
from photonpy import Dataset

# Modules
import OutputModules.output_fn as output_fn
import Model
from MinEntropyModules.generate_neighbours import KNN
import rcc



#exec(open("./setup.py").read())
#%reload_ext tensorboard

p = Path('dataset_test')
p.mkdir(exist_ok=True)

plt.close('all')

#%% functions
def load_dataset(ds, imgshape=[512, 512], shift=None):
    '''
    Loads the dataset

    Parameters
    ----------
    ds : str
        The path where the data is located.

    Returns
    -------
    locs_A , locs_B : : Nx2 nupy array
        Localizations of both channels.

    '''
    data = pd.read_csv(ds)
    grouped = data.groupby(data.Channel)
    ch1 = grouped.get_group(1)
    ch2 = grouped.get_group(2)
    
    data1 = np.array(ch1[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
    data1 = np.column_stack((data1, np.arange(data1.shape[0])))
    data2 = np.array(ch2[['X(nm)','Y(nm)', 'Pos','Int (Apert.)']])
    data2 = np.column_stack((data2, np.arange(data2.shape[0])))

    locs1 = rcc.Localizations(data1, imgshape)
    locs2 = rcc.Localizations(data2, imgshape)
    if shift is None:
        shift=locs1.align(locs2)
        print('Shifted with RCC of', shift)
        
    locs1.pos += shift
    return locs1, locs2, shift


#%% Plotting funcs
def plot_dist(locs_A, locs_B, locs_A1, locs_B1, ps=5, cmap='seismic', Filter=True, maxDist=50):
    '''
    Creates a plot as done in the paper. 

    Parameters
    ----------
    locs_A , locs_B : Localization
        Class containing all data of the channel
    locs_A1, locs_B1 : Nx2 array
        Array containing localizations of 2nd dataset.
    ps : int
        The size of the points to be plotted. The default is 10. 

    Returns
    -------
    None.

    '''
    pos_A, pos_B, dist = locs_A.couple_dataset(locs_B, Filter=Filter, maxDist=maxDist)
    pos_A1, pos_B1, dist1 = locs_A1.couple_dataset(locs_B1, Filter=Filter, maxDist=maxDist)
    
    fig, ax = plt.subplots(2,2)
    ax[0][0].scatter(pos_A[:,0], pos_A[:,1], s=ps, c=dist[:,0], cmap=cmap)
    ax[0][0].set_xlabel('x-position [nm]')
    ax[0][0].set_ylabel('Set 1 Fiducials\ny-position [nm]')
    norm=mpl.colors.Normalize(vmin=np.min(dist[:,0]), vmax=np.max(dist[:,0]), clip=False)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-offset [nmn]', ax=ax[0][0])
    
    ax[0][1].scatter(pos_A[:,0], pos_A[:,1], s=ps, c=dist[:,1], cmap=cmap)
    ax[0][1].set_xlabel('x-position [nm]')
    ax[0][1].set_ylabel('y-position [nmn]')
    norm=mpl.colors.Normalize(vmin=np.min(dist[:,1]), vmax=np.max(dist[:,1]), clip=False)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-offset [nm]', ax=ax[0][1])
    
    if dist1.shape!=(0,):
        ax[1][0].scatter(pos_A1[:,0], pos_A1[:,1], s=ps, c=dist1[:,0], cmap=cmap)
        ax[1][0].set_xlabel('x-position [nm]')
        ax[1][0].set_ylabel('Set 2 Fiducials\ny-position [nm]')
        norm=mpl.colors.Normalize(vmin=np.min(dist1[:,0]), vmax=np.max(dist1[:,0]), clip=False)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='x-offset [nm]', ax=ax[1][0])
        
        ax[1][1].scatter(pos_A1[:,0], pos_A1[:,1], s=ps, c=dist1[:,1], cmap=cmap)
        ax[1][1].set_xlabel('x-position [nm]')
        ax[1][1].set_ylabel('y-position [nm]')
        norm=mpl.colors.Normalize(vmin=np.min(dist1[:,1]), vmax=np.max(dist1[:,1]), clip=False)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='y-offset [nm]', ax=ax[1][1])
    else: print('Error: No NN found for the second dataset')


def plot_figures(locs_A, locs_B, mods, Filter=True, maxDist=50, nbins = 30, gridsize=600):
    '''
    Transforms ch2 according to the model, then couples the channels (and filters), 
    then makes a plot showing the distances between the two channels in histogram form 
    and over the full FOV

    Parameters
    ----------
    locs_A , locs_B : Localization
        Class containing all data of the channel
    mods : list
        A list containing the models used.
    Filter : bool, optional
        Filter on/off. The default is True.
    nbins : int, optional
        number of bins used in the histogram. The default is 30.
    gridsize : int, optional
        Size of the spline grids. The default is 600.

    Returns
    -------
    avg1 , avg2 : float
        The average error before and after.
    ch2_mapped : Nx2 tf Tensor
        Tensor containing the new mapped channel 2.

    '''
    # Transform dataset
    locs_B_mapped = locs_B.copy_channel()
    locs_B_mapped.pos = locs_B_mapped.transform_data(mods, gridsize=gridsize, sys_param=None)
    
    # couple datasets
    pos_A_NN, pos_B_NN,_ = locs_A.couple_dataset(locs_B, Filter=Filter, maxDist=maxDist)
    pos_A_NN_mapped, pos_B_NN_mapped,_ = locs_A.couple_dataset(locs_B_mapped, Filter=Filter, maxDist=maxDist)
    
    
    ## Plots
    # Histogram
    avg1, avg2, fig1, _ = output_fn.errorHist(pos_A_NN,  pos_B_NN, 
                                              pos_B_NN_mapped, ch1_copy2=pos_A_NN_mapped,
                                              nbins=nbins, direct=True)
    fig1.suptitle('Distribution of distances between neighbouring Localizations')
    
    # Error over FOV
    _, _, fig2, _ = output_fn.errorFOV(pos_A_NN,  pos_B_NN, 
                                       pos_B_NN_mapped, ch1_copy2=pos_A_NN_mapped,
                                       direct=True)
    fig2.suptitle('Distribution of error between neighbouring pairs over radius')
    
    return avg1, avg2, locs_B_mapped


#%% Channel Generation
## Datasets
dataset=[ # [ Path, pix_size, coupled, subset,
          #   spline gridsize, N_it, learning_rate ]
    [ [ 'C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set1/set1_beads_locs.csv' ],
     1, False, 1, 600, [20, 100], [1,1e-2] ],
    [ [ 'C:/Users/Mels/Documents/Supplementary-data/data/Registration/Set2/set2_beads_locs.csv' ],
     1, False, 1, 600, [20, 100], [1,1e-2] ]
      ]

Filter=True
maxDist=30


#%% Load 1st fiducial marker set
ds = dataset[0]
locs_A1, locs_B1, shift_rcc = load_dataset(ds[0][0])

# the data is coupled via kNN with k=1
pos_A1_NN, pos_B1_NN,_ = locs_A1.couple_dataset(locs_B1, Filter=False)
ds[2]=True

# convert to tensorflow
ch1 = tf.Variable( pos_A1_NN, dtype = tf.float32)
ch2 = tf.Variable( pos_B1_NN, dtype = tf.float32)


#%% Run optimization
mods, _ = Model.run_model(ch1, ch2, coupled=ds[2], gridsize=ds[4], N_it=ds[5],
                                   learning_rate=ds[6], plot_grid=False, pix_size=ds[1])


#%% Metrics
avg1, avg2, _ = plot_figures(locs_A1, locs_B1, mods, Filter=Filter, maxDist=maxDist, gridsize=ds[4] )    

print('\nI: The original average distance was', avg1,'. The mapping has', avg2)
print('\n Optimization 1st Sample Done')


#%% Load 2nd fiducial marker set
ds = dataset[1]
locs_A2, locs_B2, _ = load_dataset(ds[0][0], shift=shift_rcc)

#%% Metrics
avg1, avg2, locs_B_mapped = plot_figures(locs_A2, locs_B2, mods, Filter=Filter, maxDist=maxDist, gridsize=ds[4] )    

print('\nI: The original average distance was', avg1,'. The mapping has', avg2)
print('\n Optimization 2nd Sample Done')


#%% plotting everything as done in the paper
plot_dist(locs_A1, locs_B1,locs_A2, locs_B_mapped, Filter=Filter, maxDist=maxDist)
