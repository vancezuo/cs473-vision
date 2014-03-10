'''
Created on Feb 28, 2014

@author: Vance Zuo
'''

import cv2

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
        return
        
    def load_image(self, bg_path, fg_path):
        self.bg_img = cv2.imread(bg_path)
        self.fg_img = cv2.imread(fg_path)
        return True
        
    def subtract_background(self):
        # Take simple difference
        self.naive_mask = cv2.absdiff(self.bg_img, self.fg_img)
        self.naive_mask = cv2.cvtColor(self.naive_mask, cv2.COLOR_BGR2GRAY)
        __, self.naive_mask = cv2.threshold(self.naive_mask, 0, 255, 
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # MOG Subtraction
        bg_subtractor = cv2.BackgroundSubtractorMOG()
        __ = bg_subtractor.apply(self.bg_img)
        self.MOG_mask = bg_subtractor.apply(self.fg_img)
              
        # MOG2 Subtraction
        bg_subtractor = cv2.BackgroundSubtractorMOG2()
        __ = bg_subtractor.apply(self.bg_img)
        self.MOG2_mask = bg_subtractor.apply(self.fg_img)
        return
    
def main():
    obj = ObjectDetector()
    obj.load_image("example1/square.jpg", "example1/arm+square.jpg")
    obj.subtract_background()  
    cv2.imwrite("example1/naive.png", obj.naive_mask) 
    cv2.imwrite("example1/MOG.png", obj.MOG_mask)
    cv2.imwrite("example1/MOG2.png", obj.MOG2_mask)
    
if __name__ == "__main__":
    main() 