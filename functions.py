# functions.py 
'''
This Module contains the next functions
- isin_domain()
- generate_bounds()
'''
import numpy as np


def isin_domain(pos, img_param):
    '''
checks if position is in the domain 
returns True or False
    '''
    img_size = img_param.img_size_zoom()
    return pos[0] >= 0 and pos[1] >= 0 and pos[0] < img_size[0] and pos[1] < img_size[1]


def generate_bounds(img_param, pix_search = 1.5):
    '''
calculates the bounds of the search window and returns them
    '''
    pix_search = pix_search * img_param.pix_size / img_param.zoom 
    img_size = img_param.img_size_zoom()
    
    x1_lbound = int(img_size[0]/2 - pix_search)
    x1_rbound = int(img_size[0]/2 + pix_search)
    x2_lbound = int(img_size[1]/2 - pix_search)
    x2_rbound = int(img_size[1]/2 + pix_search)
    return np.array([[x1_lbound, x1_rbound] , [x2_lbound, x2_rbound] ])