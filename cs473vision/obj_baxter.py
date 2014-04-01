'''
Created on Mar 29, 2014

@author: Vance Zuo
'''

import cv2
import numpy as np
import os

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
        self.box_obj = None
        self.uncompress_obj = None
        self.compress_obj = None
        
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
    
    #TODO fine-tune default values    
    def set_arm_image(self, arm_path, hue_tolerance=180, 
                      saturation_tolerance=256, value_tolerance=96):
        if arm_path is None:
            return False
        self.arm_obj = SegmentedObject(self.bg_path, arm_path)
        arm_area = self.arm_obj.get_object_mask()
        arm_hsv = cv2.cvtColor(cv2.imread(arm_path), cv2.COLOR_BGR2HSV)
        tolerances = [hue_tolerance, saturation_tolerance, value_tolerance]
        channels = [[0], [1], [2]]
        bins = [180, 256, 256]
        ranges = [[0,179], [0,255], [0,255]]
        self.color_low = []
        self.color_high = []
        for i in range(3):
            hist = cv2.calcHist([arm_hsv], channels[i], arm_area, [bins[i]], ranges[i])
            densities = []
            for j in range(bins[i]-tolerances[i]+1):
                densities.append(sum(hist[j:j+tolerances[i]]))
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
        if self.box_obj is None:
            return (0, 0)
        return self.box_obj.get_object_rectangle()[-2:]
    
    def get_uncompressed_size(self):
        if self.uncompress_obj is None:
            return (0, 0)
        return self.uncompress_obj.get_object_rectangle()[-2:]
    
    def get_compressed_size(self):
        if self.compress_obj is None:
            return (0, 0)
        return self.compress_obj.get_object_rectangle()[-2:]
    
    def check_uncompressed_fit(self):
        return check_fit(self.get_uncompressed_size(), self.get_box_size())
    
    def check_compressed_fit(self):
        return check_fit(self.get_compressed_size(), self.get_box_size())
    
# Test script for BaxterObject
def main():
    tests = [("example5/", "no_arm.png", "arm.png"),]
    for test_param in tests:
        print "Testing:", test_param
        bg_file = test_param[0] + test_param[1]
        fg_file = test_param[0] + test_param[2]
        out_prefix = os.path.splitext(fg_file)[0]
        obj = BaxterObject(bg_file)
        obj.set_arm_image(fg_file)
        cv2.imwrite(out_prefix+"_mask.png", obj.arm_obj.get_object_mask())
        obj.arm_obj.set_ignore_color(obj.color_low, obj.color_high)
        cv2.imwrite(out_prefix+"_color.png", obj.arm_obj.color_mask)
        print obj.color_low, obj.color_high  
    return
    
if __name__ == "__main__":
    main() 