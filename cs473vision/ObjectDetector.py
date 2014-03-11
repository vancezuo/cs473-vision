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
        return True
        
    def create_fg_mask(self, method="Simple"):
        if (self.bg_img is None) or (self.fg_img is None):
            return False
        if method == "Simple":
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
            return (0, 0)
        fg_mask_copy = self.fg_mask.copy()
        contours, __ = cv2.findContours(fg_mask_copy, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        return cv2.boundingRect(contours[numpy.argmax(areas)])
    

def main():
    # Load image
    obj = ObjectDetector()
    obj.load_image("example1/square.jpg", "example1/arm+square.jpg")   
    # Test simple difference mask
    obj.create_fg_mask("Simple")  
    x, y, width, height = obj.get_object_rectangle()
    cv2.rectangle(obj.fg_mask, (x,y), (x+width,y+height), (128,128,128))
    cv2.imwrite("example1/simple.png", obj.fg_mask) 
    print "Simple difference rectangle:", obj.get_object_rectangle()
    # Test MOG mask
    obj.create_fg_mask("MOG")  
    x, y, width, height = obj.get_object_rectangle()
    cv2.rectangle(obj.fg_mask, (x,y), (x+width,y+height), (128,128,128))
    cv2.imwrite("example1/MOG.png", obj.fg_mask)
    print "MOG result rectangle:", obj.get_object_rectangle()
    # Test MOG2 mask
    obj.create_fg_mask("MOG2")  
    x, y, width, height = obj.get_object_rectangle()
    cv2.rectangle(obj.fg_mask, (x,y), (x+width,y+height), (128,128,128))
    cv2.imwrite("example1/MOG2.png", obj.fg_mask)
    print "MOG2 result rectangle:", obj.get_object_rectangle()
    return
    
if __name__ == "__main__":
    main() 
