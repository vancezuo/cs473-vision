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
        if method == "simple":
            self.fg_mask = cv2.absdiff(self.bg_img, self.fg_img)
            self.fg_mask = cv2.cvtColor(self.fg_mask, cv2.COLOR_BGR2GRAY)
            __, self.fg_mask = cv2.threshold(self.fg_mask, 0, 255,
                                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif method == "MOG":
            bg_subtractor = cv2.BackgroundSubtractorMOG()
            bg_subtractor.apply(self.bg_img)
            self.fg_mask = bg_subtractor.apply(self.fg_img)
        elif method == "MOG2":
            bg_subtractor = cv2.BackgroundSubtractorMOG2()
            bg_subtractor.apply(self.bg_img)
            self.fg_mask = bg_subtractor.apply(self.fg_img)
            __, self.fg_mask = cv2.threshold(self.fg_mask, 128, 255, 
                                             cv2.THRESH_BINARY)
        else:
            return False
        return True
    
    def get_object_rectangle(self):
        if self.fg_mask is None:
            return (0, 0, 0, 0) 
        fg_mask_copy = self.fg_mask.copy()
        contours, __ = cv2.findContours(fg_mask_copy, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return (0, 0, 0, 0)
        return cv2.boundingRect(contours[numpy.argmax(areas)])
    
def main():
    tests = [("example1/square.jpg", "example1/arm.jpg", "example1/"),
             ("example2/bg.jpg", "example2/fg1.jpg", "example2/fg1_"),
             ("example2/bg.jpg", "example2/fg2.jpg", "example2/fg2_"), 
             ("example3/bg.jpg", "example3/fg-nickel.jpg", "example3/fg-nickel_"),
             ("example3/bg.jpg", "example3/fg-penny.jpg", "example3/fg-penny_"), 
             ("example3/bg.jpg", "example3/fg-penny12.jpg", "example3/fg-penny12_")]
    obj = ObjectDetector()
    for test_param in tests:
        obj.load_image(test_param[0], test_param[1])
        print "Testing:", test_param
        for method in ["simple", "MOG", "MOG2"]:
            obj.create_fg_mask(method)  
            x, y, width, height = obj.get_object_rectangle()
            cv2.rectangle(obj.fg_mask, (x,y), (x+width,y+height), (128,128,128))
            cv2.imwrite(test_param[2] + method + ".png", obj.fg_mask) 
            print "Rectangle (" + method + "):", obj.get_object_rectangle()   
    return
    
if __name__ == "__main__":
    main() 
