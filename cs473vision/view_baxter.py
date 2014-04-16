'''
Created on Apr 15, 2014

@author: Vance Zuo
'''

import cv2
import numpy as np
import os

from obj_detect import SegmentedObject
from obj_baxter import BaxterObject

class BaxterObjectView(BaxterObject):
    def __init__(self, bg_file):
        super(BaxterObjectView, self).__init__(bg_file)
        self.name = "BaxterObject"
        self.bar = "Image"
        
        self.pos = 0
        self.total = 1
        self.seg = 0 # 0 = none, 1 = region, 2 = object
        self.rect = 0 # 0 = none, 1 = upright, 2 = min area
        return
    
    def export_results(self, ouput_dir, segment=True, roi=False, table=True):
        if segment:
            obj.export_measure_segment(ouput_dir+"measure-seg.png")
            obj.export_arm_segment(ouput_dir+"arm-seg.png")
            obj.export_uncompressed_segment(ouput_dir+"obj-seg.png")
            obj.export_compress_segment(ouput_dir+"objc-seg.png")      
        if roi:
            obj.export_measure_roi_segment(ouput_dir+"measure-roi.png")
            obj.export_arm_roi_segment(ouput_dir+"arm-roi.png")
            obj.export_uncompressed_roi_segment(ouput_dir+"obj-roi.png")
            obj.export_compress_roi_segment(ouput_dir+"objc-roi.png")
        if table:
            obj.export_sizes(example8[0] + "sizes.txt")
        return
    
    def display_results(self):
        self.total = 5 + len(self.compress_obj)
        
        cv2.destroyWindow(self.name)
        cv2.namedWindow(self.name)
        cv2.cv.CreateTrackbar(self.bar, self.name, 0, 
                              self.total, self.display_update)
        self.display_update(self.pos)
        
        while True:
            k = cv2.waitKey()
            self.pos = cv2.getTrackbarPos(self.bar, self.name)
            if k == 27 or k == ord('q'): # ESC
                break
            if k == 9: # tab
                self.display_update(0)
                cv2.waitKey(750)
            elif k == ord('a'): # left arrow
                self.pos = (self.pos - 1) % (self.total + 1)
                cv2.setTrackbarPos(self.bar, self.name, self.pos)
            elif k == ord('d'): # right arrow
                self.pos = (self.pos + 1) % (self.total + 1)
                cv2.setTrackbarPos(self.bar, self.name, self.pos)
            elif k == ord('s'):
                self.seg = (self.seg + 1) % 3
            elif k == ord('r'):
                self.rect = (self.rect + 1) % 3
            else:
                continue
            self.display_update(self.pos)
            
        cv2.destroyAllWindows()
        return
                 
    def display_update(self, index):
        bg_img = cv2.imread(self.bg_path)
        if index == 0:
            cv2.imshow(self.name, bg_img)
            return
        
        obj = None
        if index == 1:
            obj = self.measure_obj
        elif index == 2:
            obj = self.arm_obj
        elif index == 3:
            obj = self.box_obj
        elif index == 4:
            obj = self.uncompress_obj
        elif index >= 5 and index-5 < len(self.compress_obj):
            obj = self.compress_obj[index-5]
        
        if obj is None:
            black_img = np.zeros(bg_img.shape[:-1], np.uint8)
            cv2.imshow(self.name, black_img)
            return
             
        if self.seg == 2:
            obj_mask = obj.get_object_mask()
            img = cv2.bitwise_and(obj.fg_img, obj.fg_img, mask=obj_mask)
        elif self.seg == 1:
            region_mask = cv2.bitwise_and(obj.rect_mask, obj.color_mask)
            img = cv2.bitwise_and(obj.fg_img, obj.fg_img, mask=region_mask)
        else:
            img = obj.fg_img.copy()
            
        if self.rect >= 1:
            points = np.int0(obj.get_object_rectangle_points(self.rect == 2))
            cv2.drawContours(img, [points], 0, (255,255,255), 2)
            
        cv2.imshow(self.name, img)
        return

def main():
    example8 = ["example8/", "bg.jpg", "ref.jpg", (100,100), False, "arm.jpg", "obj.jpg", "obj-c.jpg", (15,30),(30,40)]
 
    bg_file = example8[0] + example8[1]
    ref_file = example8[0] + example8[2]
    width_mm, height_mm = example8[3]
    box_file = example8[0] + example8[4] if example8[4] else None
    arm_file = example8[0] + example8[5]
    obj_file = example8[0] + example8[6]
    both_file = example8[0] + example8[7]
    x_roi, y_roi = example8[8]
    w_roi, h_roi = example8[9]
     
    obj = BaxterObjectView(bg_file)
    obj.set_measure_image(ref_file, width_mm, height_mm)
    obj.set_measure_roi(x_roi, y_roi, w_roi, h_roi, "relative", "relative")
    obj.set_arm_image(arm_file)
    obj.set_arm_roi(x_roi, y_roi, w_roi, h_roi, "relative", "relative")
    obj.set_uncompressed_image(obj_file)
    obj.set_uncompressed_roi(x_roi, y_roi, w_roi, h_roi, "relative", "relative")
    obj.set_compressed_image(both_file)
    obj.set_compressed_roi(x_roi, y_roi, w_roi, h_roi, "relative", "relative")
    
#     base = "example9/foam/"
#     obj = BaxterObjectView(None)
#     obj.import_images(base)
      
    print "Color range:", obj._color_low, obj._color_high  
    print "Millimeters / Pixel:", obj.get_mm_per_px()
    print "Measure size:", obj.get_measure_size()
    print "Box size:", obj.get_box_size()
    print "Object size:", obj.get_uncompressed_size()
    print "Compressed size:", obj.get_compressed_size()
    
    obj.display_results()
    return

if __name__ == "__main__":
    main() 