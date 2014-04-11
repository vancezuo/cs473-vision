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
        self.box_size = None
        self.uncompress_obj = None
        self.arm_obj = None
        self.color_tol = None
        self.color_low = None
        self.color_high = None
        self.compress_obj = None
        self.roi = None
        
        self.set_box_image(box_path)
        self.set_uncompressed_image(obj_path)
        self.set_arm_image(arm_path)
        self.set_compressed_image(compressed_path)
       
    def export_box_segment(self, output_path):
        if self.box_obj is None:
            return False
        self.box_obj.export_object_segment(output_path)
        return True 
    
    def export_arm_segment(self, output_path):
        if self.arm_obj is None:
            return False
        self.arm_obj.export_object_segment(output_path)
        return True
    
    def export_region_segment(self, output_path):
        if self.color_low is None or self.color_high is None:
            return False
        self.compress_obj.export_region_segment(output_path)
        return True
        
    def export_uncompressed_segment(self, output_path):
        if self.uncompress_obj is None:
            return False
        self.uncompress_obj.export_object_segment(output_path)
        return True
    
    def export_compress_segment(self, output_path):
        if self.compress_obj is None:
            return False
        self.compress_obj.export_object_segment(output_path)
        return True
    
    def set_box_dimensions(self, width, height):
        self.box_size = (width, height)
        return True
    
    def set_box_image(self, box_path):
        if box_path is None:
            return False
        self.box_obj = SegmentedObject(self.bg_path, box_path)
        if not self.roi is None:
            self.box_obj.set_rectangle(*self.roi) 
        self.box_size = self.box_obj.get_object_rectangle()[-2:]
        return True
    
    def set_box_roi(self, x, y, w, h, xy_type="absolute", dim_type="absolute"):
        if self.box_obj is None:
            return False
        rect = self._get_roi(self.box_obj, x, y, w, h, xy_type, dim_type)
        self.box_obj.set_rectangle(*rect)
        return True
    
    def set_arm_color(self, color_low, color_high):
        if len(color_low) != 3 or len(color_high) != 3:
            return False
        self.color_low = color_low
        self.color_high = color_high
        return True
    
    #TODO fine-tune default values    
    def set_arm_image(self, arm_path, hue_tolerance=30, 
                      saturation_tolerance=256, value_tolerance=256):
        if arm_path is None:
            return False
        if not (0 <= hue_tolerance <= 180):
            return False
        if not (0 <= saturation_tolerance <= 256):
            return False
        if not (0 <= value_tolerance <= 256):
            return False
        self.arm_obj = SegmentedObject(self.bg_path, arm_path)      
        self.color_tol = [hue_tolerance, saturation_tolerance, value_tolerance]
        self._update_arm_color()
        return True
    
    def set_arm_roi(self, x, y, w, h, xy_type="absolute", dim_type="absolute"):
        if self.arm_obj is None:
            return False
        rect = self._get_roi(self.arm_obj, x, y, w, h, xy_type, dim_type)
        self.arm_obj.set_rectangle(*rect)
        self._update_arm_color()
        return True
   
    def set_uncompressed_image(self, uncompressed_path):
        if uncompressed_path is None:
            return False
        self.uncompress_obj = SegmentedObject(self.bg_path, uncompressed_path)
        if not self.roi is None:
            self.uncompress_obj.set_rectangle(*self.roi) 
        return True
    
    def set_uncompressed_roi(self, x, y, w, h, xy_type="absolute", 
                             dim_type="absolute"):
        if self.uncompress_obj is None:
            return False
        rect = self._get_roi(self.uncompress_obj, x, y, w, h, xy_type, dim_type)
        self.uncompress_obj.set_rect(*rect)
        return True
            
    def set_compressed_image(self, compressed_path):
        if compressed_path is None:
            return False
        self.compress_obj = SegmentedObject(self.bg_path, compressed_path)
        if not self.color_low is None and not self.color_high is None:
            self.compress_obj.set_ignore_color(self.color_low, self.color_high)
        return True
    
    def set_compressed_roi(self, x, y, w, h, xy_type="absolute", 
                             dim_type="absolute"):
        if self.compress_obj is None:
            return False
        rect = self._get_roi(self.compress_obj, x, y, w, h, xy_type, dim_type)
        self.compress_obj.set_rectangle(*rect)
        return True    
    
    def get_box_size(self):
        if self.box_size is None:
            return (0, 0)
        return self.box_size
    
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

    def _update_arm_color(self):
        arm_area = self.arm_obj.get_object_mask()
        arm_hsv = cv2.cvtColor(self.arm_obj.fg_img, cv2.COLOR_BGR2HSV)
        tolerances = self.color_tol
        channels = [[0], [1], [2]]
        bins = [180, 256, 256]
        ranges = [[0,179], [0,255], [0,255]]
        self.color_low = []
        self.color_high = []
        for i in range(3):
            hist = cv2.calcHist([arm_hsv], channels[i], arm_area, [bins[i]], ranges[i])
            densities = []
            for j in range(bins[i] - tolerances[i] + 1):
                densities.append(sum(hist[j : j+tolerances[i]]))
            min_value = np.argmax(densities)
            self.color_low.append(min_value)
            self.color_high.append(min_value + tolerances[i])
            # Debug
            #np.set_printoptions(suppress=True)
            #print hist
            #print densities
        return
    
    def _get_roi(self, ref_obj, x, y, w, h, xy_type, dim_type):
        height, width, __ = ref_obj.fg_img.shape
        if xy_type.lower() == "relative":
            x = x*width / 100
            y = y*height / 100
        if dim_type.lower() == "relative":
            w = w*width / 100
            h = h*height / 100
        if x < 0 or y < 0 or w < 0 or h < 0 or x+w > width or y+h > height:
            return (0 , 0, height, width) 
        return (x, y, w, h)
    
# Test script for BaxterObject
def main():
    # Test arm color subtraction
#     example5 = ("example5/", "no_arm.png", "arm.png")
#     print "Testing:", example5
#     bg_file = example5[0] + example5[1]
#     fg_file = example5[0] + example5[2]
#     out_prefix = os.path.splitext(fg_file)[0]
#     obj = BaxterObject(bg_file)
#     obj.set_arm_image(fg_file)
#     cv2.imwrite(out_prefix+"_mask.png", obj.arm_obj.get_object_mask())
#     obj.arm_obj.set_ignore_color(obj.color_low, obj.color_high)
#     cv2.imwrite(out_prefix+"_color.png", obj.arm_obj.color_mask)
#     print "Color range:", obj.color_low, obj.color_high  

    example6 = [("example6/w-cloth-arm/", "bg.png", "obj.png", False, "arm-cloth.png", "obj-arm-cloth.png"),
                ("example6/wo-cloth-arm/", "bg.png", "obj.png", False, "arm.png", "obj-arm.png"),]
    for test_param in example6:
        print "Testing:", test_param
        
        bg_file = test_param[0] + test_param[1]
        obj_file = test_param[0] + test_param[2]
        box_file = test_param[0] + test_param[3] if test_param[3] else None
        arm_file = test_param[0] + test_param[4] if test_param[4] else None
        both_file = test_param[0] + test_param[5] if test_param[5] else None
        
        obj = BaxterObject(bg_file)
        obj.set_uncompressed_image(obj_file)
        obj.set_box_image(box_file)
        obj.set_arm_image(arm_file)
        obj.set_compressed_image(both_file)
        obj.set_compressed_roi(0, 0, 50, 50, dim_type="relative")
        
        print "Color range:", obj.color_low, obj.color_high  
        print "Box size:", obj.get_box_size()
        print "Object size:", obj.get_uncompressed_size()
        print "Compressed size:", obj.get_compressed_size()
         
        out_prefix = os.path.splitext(obj_file)[0]
        obj.export_uncompressed_segment(out_prefix+"_segment.png")
        if not box_file is None: 
            out_prefix = os.path.splitext(box_file)[0]
            obj.export_box_segment(out_prefix+"_segment.png")
        if not arm_file is None: 
            out_prefix = os.path.splitext(arm_file)[0]
            obj.export_arm_segment(out_prefix+"_segment.png")
        if not both_file is None:     
            out_prefix = os.path.splitext(both_file)[0]
            obj.export_region_segment(out_prefix+"_roi.png")
            obj.export_compress_segment(out_prefix+"_segment.png")
        
        print     
    return
    
if __name__ == "__main__":
    main() 
