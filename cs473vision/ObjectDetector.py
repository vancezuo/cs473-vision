'''
Created on Feb 28, 2014

@author: Vance Zuo
'''

import numpy
import cv2
import os

class ObjectDetector(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.bg_img = None
        self.fg_img = None
        self.fg_mask = None
        self.ignore_mask = None
        return
        
    def load_image(self, bg_path, fg_path):
        if not (os.path.isfile(bg_path) and os.path.isfile(fg_path)):
            return False
        self.bg_img = cv2.imread(bg_path)
        self.fg_img = cv2.imread(fg_path)
        #self.bg_img = cv2.medianBlur(self.bg_img, 9)
        #self.fg_img = cv2.medianBlur(self.fg_img, 9)
        self.bg_img = cv2.bilateralFilter(self.bg_img, 5, 100, 100)
        self.fg_img = cv2.bilateralFilter(self.fg_img, 5, 100, 100)
        return True
        
    def create_fg_mask(self, method="simple"):
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

    def ignore_fg_color(self, color_min, color_max, color_type="hsv"):
        if self.fg_mask is None:
            return False
        color_min = numpy.asarray(color_min)
        color_max = numpy.asarray(color_max)
        if color_type.lower() != "hsv":
            if color_type.lower() == "rgb":
                conversion_type = cv2.COLOR_RGB2HSV
            elif color_type.lower() == "bgr":
                conversion_type = cv2.COLOR_BGR2HSV
            else:
                return False
            color_min = self.convert_color_space(color_min, conversion_type)
            color_max = self.convert_color_space(color_max, conversion_type)       
        fg_img_hsv = cv2.cvtColor(self.fg_img, cv2.COLOR_BGR2HSV)
        self.ignore_mask = cv2.inRange(fg_img_hsv, color_min, color_max)
        #self.fg_mask = cv2.bitwise_and(self.fg_mask, 
        #                               cv2.bitwise_not(self.ignore_mask))
        return True
    
    def convert_color_space(self, color, conversion_type):
        color = numpy.asarray(color)
        temp = numpy.uint8([[color]]) # 1x1 pixel "image"
        temp = cv2.cvtColor(temp, conversion_type)
        return temp[0][0]
        
    def get_object_rectangle(self):
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
