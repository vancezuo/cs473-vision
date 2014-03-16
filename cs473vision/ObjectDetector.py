'''
Created on Feb 28, 2014

@author: Vance Zuo
'''

import numpy
import cv2
import os

class ObjectDetector(object):
    '''
    An ObjectDetector attempts to detect and segment a objects from an image,
    based on a reference background image.
    
    Attributes:
        bg_img: Background image that does not contain object.
        fg_img: Foreground image of the same area as bg_img, but containing
                the object for detection.
        fg_mask: Foreground mask, with white pixels representing hypothesized 
                 foreground and black pixels representing definite background.
        ignore_mask: Foreground ignore mask where white pixels represent areas 
                     to treat automatically as background. This can be used to 
                     prevent arms from being treated as part of the foreground 
                     object.
    '''
    
    def __init__(self):
        '''
        Initiates ObjectDetector with all attributes to None.
        '''
        
        self.bg_img = None
        self.fg_img = None
        self.fg_mask = None
        self.ignore_mask = None
        return
        
    def load_image(self, bg_path, fg_path):
        '''
        Sets a new background and foreground image for object detection.
        
        Returns:
            True if both files exist and are read successfully; false otherwise.
        '''
        
        self.bg_img = cv2.imread(bg_path)
        self.fg_img = cv2.imread(fg_path)
        if (self.bg_img is None) or (self.fg_img is None):
            return False
        #self.bg_img = cv2.medianBlur(self.bg_img, 9)
        #self.fg_img = cv2.medianBlur(self.fg_img, 9)
        self.bg_img = cv2.bilateralFilter(self.bg_img, 5, 100, 100)
        self.fg_img = cv2.bilateralFilter(self.fg_img, 5, 100, 100)
        return True
        
    def create_fg_mask(self, method="simple"):
        '''
        Sets the foreground image mask to the result of a user-specified
        foreground segmentation method. 
        
        The following methods are supported:
        
            - "simple": no-frills image difference and otsu thresholding.
            - "mog": MOG background subtraction algorithm.
            - "mog2": MOG2 background subtraction algorithm.
        
        Args:
            method: The algorithm to use to create the foreground mask. Should
                    be either "simple", "mog", or "mog2".
        Returns:
            True if background and foreground images exist, a valid method
            was specified, and foreground segmentation was applied 
            successfully; false otherwise.
        '''
        
        if (self.bg_img is None) or (self.fg_img is None):
            return False
        if method.lower() == "simple":
            self.fg_mask = cv2.absdiff(self.bg_img, self.fg_img)
            self.fg_mask = cv2.cvtColor(self.fg_mask, cv2.COLOR_BGR2GRAY)
            __, self.fg_mask = cv2.threshold(self.fg_mask, 0, 255,
                                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif method.lower() == "mog":
            bg_subtractor = cv2.BackgroundSubtractorMOG()
            bg_subtractor.apply(self.bg_img)
            self.fg_mask = bg_subtractor.apply(self.fg_img)
        elif method.lower() == "mog2":
            bg_subtractor = cv2.BackgroundSubtractorMOG2()
            bg_subtractor.apply(self.bg_img)
            self.fg_mask = bg_subtractor.apply(self.fg_img)
            __, self.fg_mask = cv2.threshold(self.fg_mask, 128, 255, 
                                             cv2.THRESH_BINARY)
        else:
            return False
        return True

    def ignore_fg_color(self, color_min, color_max):
        '''
        Sets the foreground ignore mask to the all pixels falling between two
        user-specified colors in HSV (hue-saturation-value) space. 
        
        Note the colors must be in HSV space; colors specified in other spaces 
        will have undefined results.
        
        Args:
            color_min: list of length 3 containing lower bound color values 
                       in HSV space to count as part of the ignore mask.
            color_max: list of length 3 containing upper bound color values 
                       in HSV space to count as part of the ignore mask.
        Returns:
            True if ignore mask is set successfully; false otherwise.
        '''
        
        if self.fg_mask is None:
            return False
        color_min = numpy.asarray(color_min)
        color_max = numpy.asarray(color_max) 
        fg_img_hsv = cv2.cvtColor(self.fg_img, cv2.COLOR_BGR2HSV)
        self.ignore_mask = cv2.inRange(fg_img_hsv, color_min, color_max)
        #self.fg_mask = cv2.bitwise_and(self.fg_mask, 
        #                               cv2.bitwise_not(self.ignore_mask))
        return True
    
    def convert_color_space(self, color, cv2_conversion_type):
        '''
        Converts a color from one color space to another. 
        
        Args:
            color: list of length 3 representing the color to be converted.
            cv2_conversion_type: cv2 constant representing the color space
                                 conversion, e.g. cv2.COLOR_RGB2HSV for
                                 converting from RGB to HSV space.
        Returns:
            List of length 3 representing the color in the new color space.
        '''
        
        color = numpy.asarray(color)
        temp = numpy.uint8([[color]]) # 1x1 pixel "image"
        temp = cv2.cvtColor(temp, cv2_conversion_type)
        return temp[0][0]
        
    def get_object_rectangle(self):
        '''
        Returns the upright bounding rectangle of the foreground object.
        
        The object is assumed to be the largest contiguous foreground area
        according to the foreground image mask, after discounting areas marked
        as background by the ignore mask.
        
        Returns:
            A rectangle represented by the tuple (x,y,w,h), where (x,y) are the
            coordinates of the top-left corner, and (w,h) are the pixel width
            and height of the rectangle, respectively. Note if no object can 
            be identified, then (0,0,0,0) is returned.
        '''
        
        if self.fg_mask is None:
            return (0, 0, 0, 0) 
        fg_mask = self.fg_mask.copy()
        if not self.ignore_mask is None:
            fg_mask = cv2.bitwise_and(fg_mask,cv2.bitwise_not(self.ignore_mask))
        contours, __ = cv2.findContours(fg_mask, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return (0, 0, 0, 0)
        return cv2.boundingRect(contours[numpy.argmax(areas)])

# Test script for ObjectDetector
def main():
    tests = [("example1/", "square.jpg", "arm.jpg"),
             ("example2/", "bg.jpg", "fg1.jpg"),
             ("example2/", "bg.jpg", "fg2.jpg"), 
             ("example3/", "bg.jpg", "fg-nickel.jpg"),
             ("example3/", "bg.jpg", "fg-penny.jpg"), 
             ("example3/", "bg.jpg", "fg-penny12.jpg"),
             ("example4/", "bg.jpg", "fg1.jpg"),
             ("example4/", "bg.jpg", "fg2.jpg", True),]
    obj = ObjectDetector()
    for test_param in tests:
        print "Testing:", test_param
        bg_file = test_param[0] + test_param[1]
        fg_file = test_param[0] + test_param[2]
        out_prefix = os.path.splitext(fg_file)[0]
        obj.load_image(bg_file, fg_file)
        if len(test_param) > 3 and test_param[3]:
            obj.ignore_fg_color([0, 0, 0], [180, 255, 96]) # dark colors
            cv2.imwrite(out_prefix + "__ignore.png", obj.ignore_mask)
        for method in ["simple", "MOG", "MOG2"]:
            obj.create_fg_mask(method)  
            x, y, width, height = obj.get_object_rectangle()
            cv2.rectangle(obj.fg_mask, (x,y), (x+width,y+height), (128,128,128))
            cv2.imwrite(out_prefix + "_" + method + ".png", obj.fg_mask)
            print "Rectangle (" + method + "):", obj.get_object_rectangle() 
        print   
    return
    
if __name__ == "__main__":
    main() 
