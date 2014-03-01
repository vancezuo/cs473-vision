'''
Created on Feb 28, 2014

@author: Vance Zuo
'''

import numpy
import cv2

class ObjectDetector(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.bg_img = None
        self.fg_img = None
        return
        
    def load_image(self, bg_path, fg_path):
        self.bg_img = cv2.imread(bg_path)
        self.fg_img = cv2.imread(fg_path)
        return True
        
    def subtract_background(self):
        # Take simple difference
        naive = cv2.absdiff(self.bg_img, self.bg_img)
        cv2.imwrite("naive.png", naive)
        
        # MOG Subtraction
        bg_subtractor = cv2.BackgroundSubtractorMOG()
        bg_mask = bg_subtractor.apply(self.bg_img)
        fg_mask = bg_subtractor.apply(self.fg_img)
        cv2.imwrite("MOG.png", fg_mask)
        
        # MOG2 Subtraction
        bg_subtractor = cv2.BackgroundSubtractorMOG2()
        bg_mask = bg_subtractor.apply(self.bg_img)
        fg_mask = bg_subtractor.apply(self.fg_img)
        cv2.imwrite("MOG2.png", fg_mask)
        return