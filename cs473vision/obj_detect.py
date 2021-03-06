'''
Created on Feb 28, 2014

@author: Vance Zuo
'''

import argparse
import numpy as np
import cv2
import os
from math import sqrt, hypot

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
    
    def export_background(self, output_path):
        '''
        Writes the background image of the SegmentedObject to a file path
        specified by the user.
        
        Args:
            output_path: file path of output image.
        '''
        
        cv2.imwrite(output_path, self.bg_img)
        return
    
    def export_foreground(self, output_path):
        '''
        Writes the foreground image of the SegmentedObject to a file path
        specified by the user.
        
        Args:
            output_path: file path of output image.
        '''
        
        cv2.imwrite(output_path, self.fg_path)
        return
    
    def export_region_mask(self, output_path):
        '''
        Writes the region of interest of the SegmentedObject to a file path
        specified by the user. The region of interest denotes the area of the
        image considered as possible foreground (in white).
        
        Args:
            output_path: file path of output image.
        '''
        
        region_mask = cv2.bitwise_and(self.rect_mask, self.color_mask)
        cv2.imwrite(output_path, region_mask)
        return
    
    def export_region_segment(self, output_path):
        '''
        Applies the object mask of the SegmentedObject to its foreground image,
        and writes the result to a user-specified file path.
        
        Args:
            output_path: file path of output image.
        '''
        
        region_mask = cv2.bitwise_and(self.rect_mask, self.color_mask)
        segment = cv2.bitwise_and(self.fg_img, self.fg_img, mask=region_mask)
        cv2.imwrite(output_path, segment)
        return
    
    def export_object_mask(self, output_path):
        '''
        Computes the object mask of the SegmentedObject, and writes is to a 
        file path specified by the user.
        
        Args:
            output_path: file path of output image.
        '''
        
        cv2.imwrite(output_path, self.get_object_mask())
        return
    
    def export_object_segment(self, output_path, draw_rectangle=False):
        '''
        Applies the object mask of the SegmentedObject to its foreground image,
        and writes the result to a user-specified file path.
        
        Args:
            output_path: file path of output image.
        '''
        
        obj_mask = self.get_object_mask()
        segment = cv2.bitwise_and(self.fg_img, self.fg_img, mask=obj_mask)
        if draw_rectangle:
            points = self.get_object_rectangle_points()
            white = [255, 255, 255]
            for i in range(4):
                cv2.line(segment, points[i], points[(i+1)%4], white) 
        cv2.imwrite(output_path, segment)
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
    	#kernal = np.ones((7,7), np.uint8)
    	#self.fg_mask = cv2.morphologyEx(self.fg_mask, cv2.MORPH_OPEN, kernal)
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
        if color_min[0] > color_max[0]: # hue presumably "wraps" around
            color_min_upper = np.asarray([180, color_max[1], color_max[2]])
            color_max_lower = np.asarray([0, color_min[1], color_min[2]])
            mask_low = cv2.inRange(fg_img_hsv, color_max_lower, color_max)
            mask_high = cv2.inRange(fg_img_hsv, color_min, color_min_upper)
            self.color_mask = cv2.bitwise_or(mask_low, mask_high)
            # cv2.imshow("low", mask_low)
            # cv2.imshow("high", mask_high)
            # cv2.imshow("both", self.color_mask)
            # print color_max_lower, color_max
            # print color_min, color_min_upper
            # cv2.waitKey()
        else:
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
        
    def get_object_rectangle_size(self, min_area=False):
        '''
        Computes a bounding rectangle of the foreground object, and returns
        its width and height. For non-upright rectangles, the width 
        corresponds to the axis closest to x, while the height to the axis
        closest to y.
        
        Args:
            min_area: whether to compute the minimum area bounding rectangle
                      instead of a simple upright bonding rectangle.
        Returns:
            A tuple (w,h) returning the pixel width and height of the bounding
            rectangle.
        '''
        
        p1, p2, __, p4 = self.get_object_rectangle_points(min_area)
        d1 = hypot(p2[0] - p1[0], p2[1] - p1[1])
        d2 = hypot(p4[0] - p1[0], p4[1] - p1[1])
        if p2[0] - p1[0] != 0: # slope of d1 = infinity
            s1 = float(p2[1] - p1[1]) / (p2[0] - p1[0])  
            w, h = (d1, d2) if (-1 < s1 < 1) else (d2, d1)
        else:
            w, h = (d2, d1)
        return (w, h)
    
    def get_object_rectangle_points(self, min_area=False):
        '''
        Computes a bounding rectangle of the foreground object, and returns
        the xy coordinates of its 4 corners.
        
        Args:
            min_area: whether to compute the minimum area bounding rectangle
                      instead of a simple upright bonding rectangle.
        Returns:
            A 4-tuple of pairs representing the rectangle corner coordinates.
        '''
        
        contours = self._get_contours()
        areas = [cv2.contourArea(c) for c in contours]
        if not areas: # segmentation failed
            return ((0,0), (0,0), (0,0), (0,0))
        if min_area:
            min_rect = cv2.minAreaRect(contours[np.argmax(areas)])
            return cv2.cv.BoxPoints(min_rect)
        x, y, w, h = cv2.boundingRect(contours[np.argmax(areas)])
        return ((x,y), (x+w,y), (x+w,y+h), (x,y+h))
    
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
    parser = argparse.ArgumentParser(description="Segment object from background.")  
    parser.add_argument("background", help="path to background image")
    parser.add_argument("foreground", help="path to foreground image")
    parser.add_argument("-c", "--color", nargs=6, type=int, 
                        metavar=("HUE_LOW", "SAT_LOW", "VAL_LOW",
                                 "HUE_HIGH", "SAT_HIGH", "VAL_HIGH"),
                        help="specify ignored color")
    parser.add_argument("-r", "--rectangle", nargs=4, type=int,
                        metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="specify rectangle region of interest")
    parser.add_argument("-m", "--method", default="simple",
                        help="specify segmentation method")
    args = parser.parse_args()
    
    print "Importing images:", args.background+",", args.foreground
    obj = SegmentedObject(args.background, args.foreground)
    if args.color:
        color_low = [args.color[0], args.color[1], args.color[2]]
        color_high = [args.color[3], args.color[4], args.color[5]]
        print "Setting ignored color range:", color_low, "-", color_high
        obj.set_ignore_color(color_low, color_high)
    if args.rectangle:
        print "Setting rectangle region of interest:", args.rectangle
        obj.set_rectangle(args.rectangle[0], args.rectangle[1], 
                          args.rectangle[2], args.rectangle[3])
    if args.method:
        print "Attempt to use segmentation method:", args.method
        obj.set_fg_mask_method(args.method)
        
    out_prefix = os.path.splitext(args.foreground)[0]
    print "Writing result images to folder:", out_prefix
    if args.color or args.rectangle:
        obj.export_region_segment(out_prefix + "__roi.png")    
    obj.export_object_segment(out_prefix + "_" + args.method + ".png", True) 
    print "Done! Bounding rectangle size:", obj.get_object_rectangle_size() 

    return
    
if __name__ == "__main__":
    main() 
