'''
Created on Mar 29, 2014

@author: Vance Zuo
'''

import csv
import cv2
import numpy as np
import os

from obj_detect import SegmentedObject, check_fit

class BaxterObject(object):
    '''
    A BaxterObject segments and compares dimensions of objects, specifically 
    for an experimental scenario where a robot (Baxter) compares the size
    of a reference ``box'' object to a target object in uncompressed and 
    compressed forms. These objects can be imported via the constructor or
    various setter methods.
    
    The BaxterObject assumes all images are taken from a fixed location and
    distance. Based on a reference background image, it segment various objects:
    the reference ``box'' object, the uncompressed target object, and the
    robot arm. It can also segment the target object as it is being compressed, 
    but an image of the robot arm needs to set first, to allow the BaxterObject
    ignoring the arm during segmentation.
    
    Segmentation can also be limited to a rectangular area of interest. This
    is recommended if an object's location is approximately known, to ignore
    stray noise that may be mistaken as the object.
    
    Segmentation results can be output directly to the file system via a 
    series of export methods.
    
    Attributes:
        bg_path: file path to the background reference image.
        box_obj: SegmentedObject of the reference box.
        uncompress_obj: SegmentedObject of target object, uncompressed.
        arm_obj: SegmentedObject of the robot manipulating arm.
        compress_obj: list of SegmentedObject of target object, compressed 
                      (with the arm presumedly in the picture.
    '''

    def __init__(self, bg_path, box_path=None, obj_path=None, arm_path=None, 
                 compressed_path=None):
        '''
        Initiates BaxterObject with user-specified background image, and 
        optionally images of the reference box, target object, robot arm,
        and compressed object.
        
        Args:
            bg_path: file path to background image.
            box_path: (optional) file path to reference box object image.
            obj_path: (optional) file path to target object image.
            arm_path: (optional) file path to manipulator arm image. 
            compressed_path: (optional) file path to compressed target
                             object image.
        '''

        self.bg_path = bg_path
        self.box_obj = None
        self.uncompress_obj = None
        self.arm_obj = None
        self.compress_obj = []
        
        self._box_size = None # overrides box_obj for dimensions if not None
        self._color_tol = None
        self._color_low = None
        self._color_high = None
        
        self.set_box_image(box_path)
        self.set_uncompressed_image(obj_path)
        self.set_arm_image(arm_path)
        self.set_compressed_image(compressed_path)
       
    def export_box_segment(self, output_path):
        '''
        Writes a cut-out of the box, as segmented by the BaxterObject, to an
        image file. Areas not part of the segmented object are colored black.
        
        Args:
            output_path: file path of output image.
        '''
        
        if self.box_obj is None:
            return False
        self.box_obj.export_object_segment(output_path)
        return True 
    
    def export_arm_segment(self, output_path):
        '''
        Writes a cut-out of the arm, as segmented by the BaxterObject, to an
        image file. Areas not part of the segmented object are colored black.
        
        Args:
            output_path: file path of output image.
        '''
        
        if self.arm_obj is None:
            return False
        self.arm_obj.export_object_segment(output_path)
        return True
    
    def export_compress_roi_segment(self, output_path):
        '''
        Writes a cut-out of the region of interest for the compressed object
        image to an image file. Areas that would be ignored during segmentation
        are colored black.
        
        Args:
            output_path: file path of output image.
        '''
        
        if self._color_low is None or self._color_high is None:
            return False
        self.compress_obj[0].export_region_segment(output_path)
        return True
        
    def export_uncompressed_segment(self, output_path):
        '''
        Writes a cut-out of the uncompressed target object, as segmented by the 
        BaxterObject, to an image file. Areas not part of the segmented object 
        are colored black.
        
        Args:
            output_path: file path of output image.
        '''
        
        if self.uncompress_obj is None:
            return False
        self.uncompress_obj.export_object_segment(output_path)
        return True
    
    def export_compress_segment(self, output_path, all=False):
        '''
        Writes a cut-out of the ompressed target object, as segmented by the 
        BaxterObject, to an image file. Areas not part of the segmented object 
        are colored black.
        
        Args:
            output_path: file path of output image.
        '''
        
        if not self.compress_obj:
            return False
        if not all:
            all_dim = [x.get_object_rectangle()[-2:] for x in self.compress_obj]
            self.compress_obj[np.argmin(all_dim)].export_object_segment(output_path)
            return True
        for i in range(1, len(self.compress_obj)):
            cur_rect = self.compress_obj[i][-2:]
            min_rect = self.compress_obj[i]
            path_split = os.path.splitext(output_path)
            path = path_split[0] + "-" + str(i) + path_split[1]
            obj.export_object_segment(output_path)
        return True
    
    def export_sizes(self, output_path):
        with open(output_path, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["Object", "Width", "Height"])
            w, h = self.get_box_size()
            writer.writerow(["reference", w, h])
            w, h = self.get_uncompressed_size()
            writer.writerow(["uncompressed", w, h])
            count = 0
            for (w, h) in self.get_compressed_size(all=True):
                writer.writerow(["compressed-"+str(count), w, h])
                count = count + 1
        return True    
    
    def set_box_dimensions(self, width, height):
        '''
        Hard codes the pixel dimensions for the reference box object. This will 
        override any previous dimension settings, including the dimensions 
        determined by segmenting the box image.
        
        Args:
            width: pixel width to give to reference object.
            height: pixel height to give to reference object.  
        Returns:
            True.
        '''
        
        self._box_size = (width, height)
        return True
    
    def set_box_image(self, box_path):
        '''
        Loads an image as the reference box object, which will be segmented
        to determine the reference dimensions. Previous dimensions settings,
        included hardcoded dimensions, will be replaced.
        
        Args:
            box_path: file path to reference box object image. 
        Returns:
            True if image was loaded and segmented successfully, and reference
            box dimension set; false otherwise.
        '''
        
        if box_path is None:
            return False
        self.box_obj = SegmentedObject(self.bg_path, box_path)
        self._box_size = None
        return True
    
    def set_box_roi(self, x, y, w, h, xy_type="absolute", dim_type="absolute"):
        '''
        Limits to a rectangle area the region that will be processed when 
        segmenting the reference box object. Note that the parameters (x,y) 
        and (w[idth], h[eight]) can be specified in either absolute or 
        relative (to box image) terms, depending on the value of xy_type and 
        dim_type. Relative terms are treated as percentages.
        
        Note the region of interest can only be set after the box image has
        been set.
        
        Args:
            x: integer x-value of top-left point of the ROI rectangle.
            y: integer y-value of top-left point of the ROI rectangle.
            w: integer width (x-dimension) of the ROI rectangle.
            h: integer height (y-dimension) of the ROI rectangle.
            xy_type: 'absolute' if (x,y) are to be interpreted as absolute
                     pixel values; 'relative' if (x,y) are percentages of 
                     overall image from which to determine the top-left corner
                     pixel.
            dim_type: 'absolute' if (w,h) are to be interpreted as absolute
                      pixel dimensions; 'relative' if (x,y) are percentages of 
                      overall image from which to determine pixel dimensions.
        Returns:
            True if parameters represent a rectangle within in the box image;
            false otherwise.
        '''
        
        if self.box_obj is None:
            return False
        rect = self._get_roi(self.box_obj, x, y, w, h, xy_type, dim_type)
        self.box_obj.set_rectangle(*rect)
        return True
    
    def set_arm_color(self, color_low, color_high):
        '''
        Hard codes the color range of the robot arm, which will be ignored in
        the segmentation of the compressed object image. Previous color range 
        settings, including that determined from the arm image, are overriden.
        
        The colors should be in HSV space, with the domain of hue = 0 to 180,
        saturation = 0 to 256, and value = 0 to 256.
        
        Args:
            _color_low: 3-tuple denoting the lower bound HSV values of the arm.
            _color_high: 3-tuple denoting the upper bound HSV values of the arm.
        Returns:
            True if valid HSV values given and set; false otherwise.
        '''
        
        if len(color_low) != 3 or len(color_high) != 3:
            return False
        if not (0 <= color_low[0] <= 180) or not (0 <= color_high[0] <= 180):
            return False
        if not (0 <= color_low[1]  <= 256) or not (0 <= color_high[1] <= 256):
            return False
        if not (0 <= color_low[2]  <= 256) or not (0 <= color_high[2] <= 256):
            return False
        self._color_low = color_low
        self._color_high = color_high
        return True
    
    #TODO fine-tune default values    
    def set_arm_image(self, arm_path, hue_tolerance=30, 
                      saturation_tolerance=256, value_tolerance=256):
        '''
        Loads an image as the arm object, which will be segmented to determine
        its color (which will be ignored in segmentation of the compressed 
        object). Note that this color is represented as a range in HSV space,
        which is determined based on the colors the object 'mostly' consists
        of.  Previous color range settings, included hardcoded values, will 
        be replaced.
        
        The size of the range in hue, saturation, and value can be adjusted
        as parameters. Note the domain of hue = 0 to 180, saturation = 0 to 256,
        and value = 0 to 256.
        
        Args:
            arm_path: file path to arm object image. 
            hue_tolerance: size of the hue range for the arm's color.
            saturation_tolerance: size of saturation range for the arm's color.
            value_tolerance: size of the value range for the arm's color.
        Returns:
            True if image was loaded and segmented successfully, and arm
            color range set; false otherwise.
        '''
        
        if arm_path is None:
            return False
        if not (0 <= hue_tolerance <= 180):
            return False
        if not (0 <= saturation_tolerance <= 256):
            return False
        if not (0 <= value_tolerance <= 256):
            return False
        self.arm_obj = SegmentedObject(self.bg_path, arm_path)      
        self._color_tol = [hue_tolerance, saturation_tolerance, value_tolerance]
        self._update_arm_color()
        return True
    
    def set_arm_roi(self, x, y, w, h, xy_type="absolute", dim_type="absolute"):
        '''
        Limits to a rectangle area the region that will be processed when 
        segmenting the arm object, and recalculates its color range (see
        set_arm_image() for color range calculation details)).
        
        The parameters (x,y) and (w[idth], h[eight]) can be specified in 
        either absolute or relative terms. Relative terms are treated 
        as percentages of overall image size.
        
        Note the region of interest can only be set after the arm image has
        been set.
        
        Args:
            x: integer x-value of top-left point of the ROI rectangle.
            y: integer y-value of top-left point of the ROI rectangle.
            w: integer width (x-dimension) of the ROI rectangle.
            h: integer height (y-dimension) of the ROI rectangle.
            xy_type: 'absolute' if (x,y) are to be interpreted as absolute
                     pixel values; 'relative' if (x,y) are percentages of 
                     overall image from which to determine the top-left corner
                     pixel.
            dim_type: 'absolute' if (w,h) are to be interpreted as absolute
                      pixel dimensions; 'relative' if (x,y) are percentages of 
                      overall image from which to determine pixel dimensions.
        Returns:
            True if parameters represent a rectangle within in the arm image;
            false otherwise.
        '''
        
        if self.arm_obj is None:
            return False
        rect = self._get_roi(self.arm_obj, x, y, w, h, xy_type, dim_type)
        self.arm_obj.set_rectangle(*rect)
        self._update_arm_color()
        return True
   
    def set_uncompressed_image(self, uncompressed_path):
        '''
        Loads an image as the target object in uncompressed form.
        
        Args:
            uncompressed_path: file path to arm object image. 
        Returns:
            True if image was loaded and segmented successfully;
            false otherwise.
        '''
        
        if uncompressed_path is None:
            return False
        self.uncompress_obj = SegmentedObject(self.bg_path, uncompressed_path)
        return True
    
    def set_uncompressed_roi(self, x, y, w, h, xy_type="absolute", 
                             dim_type="absolute"):
        '''
        Limits to a rectangle area the region that will be processed when 
        segmenting the uncompressed target object. 
        
        Note that the parameters (x,y)  and (w[idth], h[eight]) can be 
        specified in either absolute or relative terms, depending on the value 
        of xy_type and dim_type. Relative terms are treated as percentages.
        
        Note the region of interest can only be set after the uncompressed
        object image has been set.
        
        Args:
            x: integer x-value of top-left point of the ROI rectangle.
            y: integer y-value of top-left point of the ROI rectangle.
            w: integer width (x-dimension) of the ROI rectangle.
            h: integer height (y-dimension) of the ROI rectangle.
            xy_type: 'absolute' if (x,y) are to be interpreted as absolute
                     pixel values; 'relative' if (x,y) are percentages of 
                     overall image from which to determine the top-left corner
                     pixel.
            dim_type: 'absolute' if (w,h) are to be interpreted as absolute
                      pixel dimensions; 'relative' if (x,y) are percentages of 
                      overall image from which to determine pixel dimensions.
        Returns:
            True if parameters represent a rectangle within in the object image;
            false otherwise.
        '''
        
        if self.uncompress_obj is None:
            return False
        rect = self._get_roi(self.uncompress_obj, x, y, w, h, xy_type, dim_type)
        self.uncompress_obj.set_rectangle(*rect)
        return True
            
    def set_compressed_image(self, compressed_path, add=True):
        '''
        Loads an image as the target object in compressed form. The compressing
        robot arm is assumed to also be in the image, so areas with the same
        color as it will be ignored. Thus, the compressed object should have
        a different color from the arm.
        
        Args:
            compressed_path: file path to arm object image. 
            add: boolean denoting whether to add the compressed image to the
                 list of images, or to create a new list starting with the
                 current image.
        Returns:
            True if image was loaded and segmented successfully;
            false otherwise.
        '''
        
        if compressed_path is None:
            return False
        new_obj = SegmentedObject(self.bg_path, compressed_path)
        if not self._color_low is None and not self._color_high is None:
            new_obj.set_ignore_color(self._color_low, self._color_high)
        if add:
            self.compress_obj.append(new_obj)
        else:
            self.compress_obj = [new_obj]
        return True
    
    def set_compressed_roi(self, x, y, w, h, xy_type="absolute", 
                             dim_type="absolute"):
        '''
        Limits to a rectangle area the region that will be processed when 
        segmenting the compressed target object. 
        
        Note that the parameters (x,y)  and (w[idth], h[eight]) can be 
        specified in either absolute or relative terms. 
        Relative terms are treated as percentages.
        
        Note the region of interest can only be set after the uncompressed
        object image has been set.
        
        Args:
            x: integer x-value of top-left point of the ROI rectangle.
            y: integer y-value of top-left point of the ROI rectangle.
            w: integer width (x-dimension) of the ROI rectangle.
            h: integer height (y-dimension) of the ROI rectangle.
            xy_type: 'absolute' if (x,y) are to be interpreted as absolute
                     pixel values; 'relative' if (x,y) are percentages of 
                     overall image from which to determine the top-left corner
                     pixel.
            dim_type: 'absolute' if (w,h) are to be interpreted as absolute
                      pixel dimensions; 'relative' if (x,y) are percentages of 
                      overall image from which to determine pixel dimensions.
        Returns:
            True if parameters represent a rectangle within in the object image;
            false otherwise.
        '''
        
        if not self.compress_obj:
            return False
        rect = self._get_roi(self.compress_obj[0], x, y, w, h, xy_type, dim_type)
        for obj in self.compress_obj:
            obj.set_rectangle(*rect)
        return True    
    
    def get_box_size(self):
        '''
        Returns the width and height dimensions of the reference box object.
        
        Returns:
            A pair (width, height) denoting the box's dimensions.
        '''
        
        if not self._box_size is None:
            return self._box_size
        if self.box_obj is None:
            return (0, 0)
        return self.box_obj.get_object_rectangle()[-2:]
    
    def get_uncompressed_size(self):
        '''
        Returns the width and height dimensions of the uncompressed target
        object.
        
        Returns:
            A pair (width, height) denoting the object's dimensions.
        '''
        
        if self.uncompress_obj is None:
            return (0, 0)
        return self.uncompress_obj.get_object_rectangle()[-2:]
    
    def get_compressed_size(self, all=False):
        '''
        Returns the width and height dimensions of the compressed target
        object.
        
        Args:
            all: whether or not to return all dimension from the list of
                 compressed object images.
        Returns:
            A list of pairs (width, height) denoting the object's dimensions
            at different compression instances if all is True; else just
            the pair with minimum area.
        '''
        
        if not self.compress_obj:
            return [(0, 0)]
        all_dim = [x.get_object_rectangle()[-2:] for x in self.compress_obj]
        if all:
            return all_dim
        return min(all_dim, key=(lambda x: x[0]*x[1]))
    
    def check_uncompressed_fit(self):
        '''
        Checks if the uncompressed target object 'fits' in the reference 
        box object.
        
        Returns:
            True if the uncompressed target object's dimensions 'fit' in the
            reference box object's dimensions; false otherwise.
        '''
        
        return check_fit(self.get_uncompressed_size(), self.get_box_size())
    
    def check_compressed_fit(self):
        '''
        Checks if the compressed target object 'fits' in the reference 
        box object.
        
        Returns:
            True if the compressed target object's dimensions 'fit' in the
            reference box object's dimensions; false otherwise.
        '''
        
        return check_fit(self.get_compressed_size(), self.get_box_size())

    def _update_arm_color(self):
        arm_area = self.arm_obj.get_object_mask()
        arm_hsv = cv2.cvtColor(self.arm_obj.fg_img, cv2.COLOR_BGR2HSV)
        tolerances = self._color_tol
        channels = [[0], [1], [2]]
        bins = [180, 256, 256]
        ranges = [[0,179], [0,255], [0,255]]
        self._color_low = []
        self._color_high = []
        for i in range(3):
            hist = cv2.calcHist([arm_hsv], channels[i], arm_area, [bins[i]], ranges[i])
            densities = []
            for j in range(bins[i] - tolerances[i] + 1):
                densities.append(sum(hist[j : j+tolerances[i]]))
            min_value = np.argmax(densities)
            self._color_low.append(min_value)
            self._color_high.append(min_value + tolerances[i])
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
#     obj.arm_obj.set_ignore_color(obj._color_low, obj._color_high)
#     cv2.imwrite(out_prefix+"_color.png", obj.arm_obj.color_mask)
#     print "Color range:", obj._color_low, obj._color_high  

    examples = [#("example6/w-cloth-arm/", "bg.png", "obj.png", False, "arm-cloth.png", "obj-arm-cloth.png"),
                #("example6/wo-cloth-arm/", "bg.png", "obj.png", False, "arm.png", "obj-arm.png"),
                ("example7/yellow-phone/", "bg.png", "obj.png", False, "arm-cloth.png", "obj-arm-cloth.png"),
                ("example7/blue-sq/", "bg.png", "obj.png", False, "arm-cloth.png", "obj-arm-cloth.png"),
                ("example7/brown-box/", "bg.png", "obj.png", False, "arm-cloth.png", "obj-arm-cloth.png")
                ]
    for test_param in examples:
        print "Testing:", test_param
        
        bg_file = test_param[0] + test_param[1]
        obj_file = test_param[0] + test_param[2]
        box_file = test_param[0] + test_param[3] if test_param[3] else None
        arm_file = test_param[0] + test_param[4] if test_param[4] else None
        both_file = test_param[0] + test_param[5] if test_param[5] else None
        
        obj = BaxterObject(bg_file)
        obj.set_uncompressed_image(obj_file)
        obj.set_uncompressed_roi(0, 0, 50, 50, dim_type="relative")
        obj.set_box_image(box_file)
        obj.set_arm_image(arm_file)
        obj.set_compressed_image(both_file)
        obj.set_compressed_roi(0, 0, 50, 50, dim_type="relative")
        
        print "Color range:", obj._color_low, obj._color_high  
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
            obj.export_compress_roi_segment(out_prefix+"_roi.png")
            obj.export_compress_segment(out_prefix+"_segment.png")
        obj.export_sizes(out_prefix+"sizes.txt")
        print     
    return
    
if __name__ == "__main__":
    main() 
