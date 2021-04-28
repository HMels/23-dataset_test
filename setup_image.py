# setup_image.py
"""
This scrit will contain all classes used for image parameters like image size, center image 
but also deformations


first I have to make img consistent
"""
import numpy as np

class Deform():
    '''
    This class contains all functions and variables used to give the image a deformation
    
    The variables are:
        - shift
        - rotation
        - shear
        - scaling
    
    The functions are:
        - deformation()
        - ideformation()
        - shift_def()
        - shift_idef()
        - rotation_def()
        - rotation_idef()
        - shear_def()
        - shear_idef()
        - scaling_def()
        - scaling_idef()
    '''
    
    def __init__(self, shift, rotation, shear, scaling):
        self.shift = shift
        self.rotation = rotation * 100 # The adjustments in rotation needs to be fine
        self.shear = shear
        self.scaling = scaling
        
        
    def deform(self, locs):
        if self.shift[0] != 0 and self.shift[1] != 0:
            locs = self.shift_def(locs)
        if self.rotation != 0:
            locs = self.rotation_def(locs)
        if self.shear[0] != 0 and self.shear[1] != 0:
            locs = self.shear_def(locs)
        if self.scaling[0] != 1 and self.scaling[1] != 1:
            locs = self.scaling_def(locs)
        return locs
    
    
    def ideform(self, locs):
        if self.scaling[0] != 1 and self.scaling[1] != 1:
            locs = self.scaling_idef(locs)
        if self.shear[0] != 0 and self.shear[1] != 0:
            locs = self.shear_idef(locs)
        if self.rotation != 0:
            locs = self.rotation_idef(locs)
        if self.shift[0] != 0 and self.shift[1] != 0:
            locs = self.shift_idef(locs)
        return locs
        
    
    def shift_def(self, locs):
        return locs + self.shift
    
    
    def shift_idef(self, locs):
        return locs - self.shift
    
    
    def rotation_def(self, locs):
        cos = np.cos(self.rotation / 100 * 0.0175) 
        sin = np.sin(self.rotation / 100 * 0.0175)
       
        locs = np.array([
             (cos * locs[:,0] - sin * locs[:,1]) ,
             (sin * locs[:,0] + cos * locs[:,1]) 
            ]).transpose()
        return locs
    
    
    def rotation_idef(self, locs):
        cos = np.cos(self.rotation / 100 * 0.0175) 
        sin = np.sin(self.rotation / 100 * 0.0175)
       
        locs = np.array([
             (cos * locs[:,0] + sin * locs[:,1]) ,
             (-1*sin * locs[:,0] + cos * locs[:,1]) 
            ]).transpose()
        return locs
    
    
    def shear_def(self, locs):
        locs = np.array([
            locs[:,0] + self.shear[0]*locs[:,1] ,
            self.shear[1]*locs[:,0] + locs[:,1] 
            ]).transpose()
        return locs
    
    
    def shear_idef(self, locs):
        locs = np.array([
            locs[:,0] - self.shear[0]*locs[:,1] ,
            -1*self.shear[1]*locs[:,0] + locs[:,1] 
            ]).transpose()
        return locs
    
    
    def scaling_def(self, locs):
        locs = np.array([
            self.scaling[0] * locs[:,0] ,
            self.scaling[1] * locs[:,1]
            ]).transpose()
        return locs
    
    
    def scaling_idef(self, locs):
        locs = np.array([
            (1/self.scaling[0]) * locs[:,0] ,
            (1/self.scaling[1]) * locs[:,1]
            ]).transpose()
        return locs