'''
Created on Feb 28, 2014

@author: Vance Zuo
'''

import numpy as np
import cv2
import os
from math import sqrt

class SegmentedObject(object):
    '''
    A SegmentedObject attempts to represent an object from an image,
    based on a reference background image.
    
    Attributes:
        bg_img: Background image that does not contain object.
        fg_img: Foreground image of the same area as bg_img, but containing
                the object for detection.
        fg_mask: Foreground mask, with white pixels representing hypothesized 
                 foreground and black pixels representing definite background.
        color_mask: Foreground color mask, where black pixels represent areas 
                     to treat automatically as background. This can be used to 
                     prevent arms from being treated as part of the foreground 
                     object.
        rect_mask: Foreground rectangle mask, where black pixels represent
                   areas to treat automatically as background. This can be used
                   to establish a region of focus.
    '''
    
    def __init__(self, bg_path, fg_path, method="simple", color_range=None,
                 rectangle=None):
        '''
        Initiates SegmentedObject with user-specified background and foreground
        image paths.
        
        Args:
            bg_path: file path to background image.
            fg_path: file path to foreground image.
            method: (optional) algorithm to use to create the foreground mask. 
            rectangle: (optional) 4-tuple representing the foreground rectangle
                       mask to use.
            color_range: (optional) 2-tuple representing the color range for
                         the foreground color mask.
        '''

        self.bg_img = cv2.imread(bg_path)
        if self.bg_img is None:
            raise IOError("Background image not loaded successfully.")
        self.fg_img = cv2.imread(fg_path)
        if self.fg_img is None:
            raise IOError("Foreground image not loaded successfully.")
      
        # Blurring images smooths out noise 
        self.bg_img = cv2.bilateralFilter(self.bg_img, 5, 100, 100)
        self.fg_img = cv2.bilateralFilter(self.fg_img, 5, 100, 100)
        #self.bg_img = cv2.medianBlur(self.bg_img, 9)
        #self.fg_img = cv2.medianBlur(self.fg_img, 9)
        
        # Initalizes masks
        white_mask = cv2.bitwise_not(np.zeros(self.fg_img.shape[:-1], np.uint8))
        self.rect_mask = white_mask
        if not rectangle is None:
            self.set_rectangle(*rectangle)
        self.color_mask = white_mask
        if not color_range is None:
            self.set_ignore_color(*color_range)
        self.set_fg_mask_method(method)
        return
        
    def set_fg_mask_method(self, method):
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

    def set_rectangle(self, x, y, width, height):
        '''
        Sets the foreground rectangle mask to all pixels within the bounds of a
        user-specified rectangle. 
        
        Args:
            x: the x-value of the top-left pixel of the rectangle.
            y: the y-value of the top-left pixel of the rectangle.
            width: total width of rectangle.
            height: total height of rectangle.
            
        Returns:
            True if foreground rectangle mask set successfully; false otherwise.
        '''
        
        if self.fg_img is None:
            return False
        self.rect_mask = np.zeros(self.fg_img.shape[:-1], np.uint8)
        cv2.rectangle(self.rect_mask, (x,y), (x+width,y+height), 
                      (255, 255, 255), cv2.cv.CV_FILLED)
        return True
    

    def set_ignore_color(self, color_min, color_max):
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
        
        if self.fg_img is None:
            return False
        color_min = np.asarray(color_min)
        color_max = np.asarray(color_max) 
        fg_img_hsv = cv2.cvtColor(self.fg_img, cv2.COLOR_BGR2HSV)
        #if color_min[0] > color_max[0]: # hue presumedly "wraps" around
        #    color_min_upper = [180, color_min[1], color_min[2]]
        #    color_max_lower = [0, color_max[1], color_max[2]]
        #    mask_low = cv2.inRange(fg_img_hsv, color_max_lower, color_max)
        #    mask_high = cv2.inRange(fg_img_hsv, color_min, color_min_upper)
        #    self.color_mask = cv2.bitwise_or(mask_low, mask_high)
        self.color_mask = cv2.inRange(fg_img_hsv, color_min, color_max)
        self.color_mask = cv2.bitwise_not(self.color_mask)
        return True
    
    def get_object_mask(self):
        '''
        Computes an image mask, where white represents the foreground object
        and black represents background.
        
        Returns:
            A matrix representing a 8-bit image mask, with white pixels denoting
            the object, and black pixels representing background.
        '''
        
        contours = self._get_contours()
        areas = [cv2.contourArea(c) for c in contours]
        object_mask = np.zeros(self.fg_mask.shape, np.uint8)
        if not areas:
            return object_mask
        cv2.drawContours(object_mask, contours, np.argmax(areas), 
                         (255,255,255), cv2.cv.CV_FILLED)
        return object_mask
        
    def get_object_rectangle(self):
        '''
        Computes the upright bounding rectangle of the foreground object.
        
        Returns:
            A rectangle represented by the tuple (x,y,w,h), where (x,y) are the
            coordinates of the top-left corner, and (w,h) are the pixel width
            and height of the rectangle, respectively. Note if no object can 
            be identified, then (0,0,0,0) is returned.
        '''
        
        contours = self._get_contours()
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return (0, 0, 0, 0)
        return cv2.boundingRect(contours[np.argmax(areas)])
    
    def _get_contours(self):
        '''
        Helper method for extracting contours and contour areas of possible
        objects in the foreground image. Not to be used by user.
        
        The object is assumed to be the largest contiguous foreground area
        according to the foreground image mask, after discounting areas marked
        as background by the ignore mask and rectangle mask.
        
        Returns:
            List of lists of points representing detected contours in 
            foreground mask.
        '''
        
        if self.fg_mask is None:
            return None
        fg_mask = self.fg_mask.copy()
        if not self.color_mask is None:
            fg_mask = cv2.bitwise_and(fg_mask, self.color_mask)
        if not self.rect_mask is None:
            fg_mask = cv2.bitwise_and(fg_mask, self.rect_mask)
        contours, __ = cv2.findContours(fg_mask, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        return contours 
    
def check_fit((w1, h1), (w2, h2)):
    '''
    Checks if a rectangle 'fits' inside another rectangle.
    
    Args:
        w1: width of the rectangle of interest.
        h1: height of the rectangle of interest.
        w2: width of the rectangle to compare to.
        h1: height of the rectangle to compare to.
    Returns:
        True if rect1's dimensions fit within rect2's dimensions; 
        false otherwise.
    '''  
    
    w1, h1 = (w1, h1) if w1 >= h1 else (h1, w1) # need width >= height
    w2, h2 = (w2, h2) if w2 >= h2 else (h2, w2)
    return ((w1 <= w2 and h1 <= h2) or 
            (w1 > w2 and h2 >= float(2*w1*h1*w2 + (w1*w1 - h1*h1)*sqrt(w1*w1 + h1*h1 - w2*w2)) / (w1*w1 + h1*h1)))
    
def convert_color_space(color, cv2_conversion_type):
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
    
    color = np.asarray(color)
    temp = np.uint8([[color]]) # 1x1 pixel "image"
    temp = cv2.cvtColor(temp, cv2_conversion_type)
    return temp[0][0]

# Test script for SegmentedObject
def main():
    tests = [("example1/", "square.jpg", "arm.jpg"),
             ("example2/", "bg.jpg", "fg1.jpg"),
             ("example2/", "bg.jpg", "fg2.jpg"), 
             ("example3/", "bg.jpg", "fg-nickel.jpg"),
             ("example3/", "bg.jpg", "fg-penny.jpg"), 
             ("example3/", "bg.jpg", "fg-penny12.jpg"),
             ("example4/", "bg.jpg", "fg1.jpg"),
             ("example4/", "bg.jpg", "fg2.jpg", True),]
    for test_param in tests:
        print "Testing:", test_param
        bg_file = test_param[0] + test_param[1]
        fg_file = test_param[0] + test_param[2]
        out_prefix = os.path.splitext(fg_file)[0]
        obj = SegmentedObject(bg_file, fg_file)
        if len(test_param) > 3 and test_param[3]:
            obj.set_ignore_color([0, 0, 0], [180, 255, 96]) # dark colors
            obj.set_rectangle(450, 100, 300, 225) # random region of focus
            ignore_mask = cv2.bitwise_and(obj.color_mask, obj.rect_mask)
            cv2.imwrite(out_prefix + "__ignore.png", ignore_mask)     
        for method in ["simple", "MOG", "MOG2"]:
            obj.set_fg_mask_method(method)  
            x, y, width, height = obj.get_object_rectangle()
            cv2.rectangle(obj.fg_mask, (x,y), (x+width,y+height), (128,128,128))
            cv2.imwrite(out_prefix + "_" + method + ".png", obj.fg_mask) 
            print "Rectangle (" + method + "):", obj.get_object_rectangle() 
        print   
    return
    
if __name__ == "__main__":
    main() 
