'''
Created on Mar 29, 2014

@author: Vance Zuo
'''

import cv2
import numpy as np

from obj_detect import SegmentedObject, check_fit

class BaxterObject(object):
    '''
    classdocs
    '''

    def __init__(self, bg_path, box_path=None, obj_path=None, arm_path=None, 
                 compressed_path=None):
        '''
        Constructor
        '''
        self.bg_path = bg_path
        self.set_box_image(box_path)
        self.set_uncompressed_image(obj_path)
        self.set_arm_image(arm_path)
        self.set_compressed_image(compressed_path)
    
    def set_box_image(self, box_path):
        if box_path is None:
            return False
        self.box_obj = SegmentedObject(self.bg_path, box_path)
        return True
    
    def set_uncompressed_image(self, uncompressed_path):
        if uncompressed_path is None:
            return False
        self.uncompress_obj = SegmentedObject(self.bg_path, uncompressed_path)
        return True
        
    def set_arm_image(self, arm_path, hue_tolerance=180, #TODO fine-tune default values
                      saturation_tolerance=256, value_tolerance=256):
        if arm_path is None:
            return False
        arm_area = SegmentedObject(self.bg_path, arm_path).get_object_mask()
        arm_hsv = cv2.cvtColor(cv2.imread(arm_path), cv2.COLOR_BGR2HSV)
        tolerances = [hue_tolerance, saturation_tolerance, value_tolerance]
        channels = [[0], [1], [2]]
        bins = [180, 256, 256]
        ranges = [[0,180], [0,256], [0,256]]
        self.color_low = []
        self.color_high = []
        for i in range(3):
            hist = cv2.calcHist([arm_hsv], channels[i], arm_area, bins[i], ranges[i])
            densities = [sum(hist[j:tolerances[i]]) for j in range(bins[i]-tolerances[i])]
            min_value = np.argmin(densities)
            self.color_low.append(min_value)
            self.color_high.append(min_value + tolerances[i])
        return True
        
    def set_compressed_image(self, compressed_path):
        if compressed_path is None:
            return False
        self.compress_obj = SegmentedObject(self.bg_path, compressed_path)
        if not self.color_low is None and not self.color_high is None:
            self.compress_obj.set_ignore_color(self.color_low, self.color_high) 
        return True
    
    def get_box_size(self):
        return self.box_obj.get_object_rectangle()[-2:]
    
    def get_uncompressed_size(self):
        return self.uncompress_obj.get_object_rectangle()[-2:]
    
    def get_compressed_size(self):
        return self.compress_obj.get_object_rectangle()[-2:]
    
    def check_uncompressed_fit(self):
        return check_fit(self.get_uncompressed_size(), self.get_box_size())
    
    def check_compressed_fit(self):
        return check_fit(self.get_compressed_size(), self.get_box_size())