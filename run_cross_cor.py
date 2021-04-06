# run_cross_cor.py
'''
This file contains all code to run the cross correlation program on a certain dataset.
It uses the Monte Carlo Method for N_it iterations for different values of zoom
to create an array containing the average error per zoom value and its std
'''

#%% Monte Carlo Simulation for accuracy via cross_correlation
zoom = np.array([1, 2, 3, 4, 6, 9, 12, 15])
N_it = 40
shift_array = np.zeros([len(zoom), N_it])
    
for z in range(len(zoom)):
    img_param.zoom = zoom[z]
    angle = angle0 * np.pi / 180          # angle in radians
    shift = shift0 / img_param.zoom      # shift in units of system [zoom]
    for i in range(N_it):
        channel_A, channel_B, _, _ = run_channel_generation(cluster, img_param, 
                                                            angle, shift, error = 0.1)
        _ , abs_error_nm = cross_correlation.cross_corr_script(channel_A, channel_B, 
                                                               img_param, shift, pix_search = 1.5,
                                                               output_on = False)
        shift_array[z,i] = abs_error_nm

accuracy_avg = np.sum(shift_array, 1) / N_it
accuracy_std = np.std(shift_array, 1)
    
plt.close('all')
plt.figure()
eb1 = plt.errorbar(zoom, accuracy_avg, yerr = accuracy_std, lw=1.5, ls=':')
#eb1[-1][0].set_linestyle('--')
plt.xlabel('Precission [nm]')
plt.ylabel('Accuracy [nm]')
plt.ylim(0)
plt.xlim(0)
    
plt.axhline(10, ls = '--', lw = 0.5, label = '10nm')
plt.axhline(5, ls = '--', lw = 0.5, label = '5nm')
plt.legend()