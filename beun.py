# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:37:25 2021

@author: Mels
"""
import sys
sys.path.append('../')

from photonpy import Dataset
import numpy as np
from LoadDataModules.load_data import *
import OutputModules.output_fn as output_fn
from MinEntropyModules.generate_neighbours import KNN
import matplotlib.pyplot as plt
import OutputModules.generate_image as generate_image


subset=0.2
path = [ 'C:/Users/Mels/Documents/example_MEP/ch0_locs.hdf5' , 'C:/Users/Mels/Documents/example_MEP/ch1_locs.hdf5' ]

ch1 = Dataset.load(path[0])
ch2 = Dataset.load(path[1])
Dataset.align(ch1, ch2)
ch1_data = ch1.pos
ch2_data = ch2.pos
ch1_data[:,[0, 1]] = ch1_data[:,[1, 0]]*10
ch2_data[:,[0, 1]] = ch2_data[:,[1, 0]]*10

img = np.empty([2,2], dtype = float)        # calculate borders of system
img[0,0] = np.min(ch1_data[:,0])
img[1,0] = np.max(ch1_data[:,0])
img[0,1] = np.min(ch1_data[:,1])
img[1,1] = np.max(ch1_data[:,1])
size_img = img[1,:] - img[0,:]
mid = (img[1,:] + img[0,:])/2

ch1_data -= mid
ch2_data -= mid

l_grid = -np.array([ subset*size_img[0], subset*size_img[1] ])/2
r_grid = np.array([ subset*size_img[0], subset*size_img[1] ])/2
    
indx1 = np.argwhere( (ch1_data[:,0] >= l_grid[0]) * (ch1_data[:,1] >= l_grid[1])
                    * (ch1_data[:,0] <= r_grid[0]) * (ch1_data[:,1] <= r_grid[1]) )[:,0]
    
indx2 = np.argwhere( (ch2_data[:,0] >= l_grid[0]) * (ch2_data[:,1] >= l_grid[1])
                    * (ch2_data[:,0] <= r_grid[0]) * (ch2_data[:,1] <= r_grid[1]) )[:,0]

indx = np.arange(10000,20000)
indx1 = indx
indx2 = indx
ch1 = ch1_data[ indx1, : ]
ch2 = ch2_data[ indx2, : ]

r = np.sqrt(np.sum(ch1**2,1))
idx1 = KNN(ch1, ch2, 1)
error1 = np.sqrt( np.sum( ( ch1 - ch2[idx1,:][:,0] )**2, axis = 1) )
error2 = ch1 - ch2[idx1,:][:,0,:]
avg1 = np.average(error1)

#%% plot
plt.close('all')
plt.figure()
plt.plot(r, error1, 'b.', alpha=.4, label='Original error')
xmax= np.max(r)+50
plt.hlines(avg1, color='purple', xmin=0, xmax=xmax, label=('average original = '+str(round(avg1,2))))

plt.ylim(0)
plt.xlim([0,xmax])
plt.xlabel('FOV [nm]')
plt.ylabel('Absolute Error')

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(r, error2[:,0], 'b.', alpha=.4, label='Original error')
ax1.set_xlabel('x1 [nm]')
ax1.set_ylabel('x1 error [nm]')
ax2.plot(r, error2[:,1], 'b.', alpha=.4, label='Original error')
ax2.set_xlabel('x2 [nm]')
ax2.set_ylabel('x2 error [nm]')

#%% plot img
## Channel Generation
channel1, channel2, channel2m, bounds = generate_image.generate_channel(
        ch1, ch2, ch2, 1)
generate_image.plot_1channel(channel1, bounds, 1)
