'''
Created on Apr 15, 2014

@author: Vance Zuo
'''

import argparse
import cv2
import numpy as np
import os

from obj_detect import SegmentedObject
from obj_baxter import BaxterObject

class BaxterExperiment(BaxterObject):
    def __init__(self, bg_file=None):
        super(BaxterExperiment, self).__init__(bg_file)
        self.name = "BaxterObject"
        self.bar = "Image"
        
        self.pos = 0
        self.total = 1
        self.seg = 0 # 0 = none, 1 = region, 2 = object
        self.rect = 0 # 0 = none, 1 = upright, 2 = min area
        return
    
    def export_results(self, ouput_dir, segment=True, roi=False, table=True):
        if not os.path.isdir(ouput_dir):
            return False
        if not ouput_dir.endswith("/"):
            ouput_dir += "/"
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
        return True
    
    def import_images(self, path_dir): # Caution: very project specific
        if not os.path.isdir(path_dir):
            return False
        if not path_dir.endswith("/"):
            path_dir += "/"
        for file in os.listdir(path_dir): # Must find background first
            if file.endswith(".png") or file.endswith(".jpg"):
                name = os.path.splitext(file)[0]
                if name == "background" or name == "bg":
                    self.bg_path = path_dir + file
                    break
        if not self.bg_path:
            return False
        for file in os.listdir(path_dir):
            if file.endswith(".png") or file.endswith(".jpg"):
                name = os.path.splitext(file)[0]
                if name == "reference" or name == "ref":
                    self.set_measure_image(path_dir + file, 100, 100)
                elif name == "arm":
                    self.set_arm_image(path_dir + file)
                elif name == "box":
                    self.set_box_image(path_dir + file)
                elif name == "object" or name == "obj":
                    self.set_uncompressed_image(path_dir + file)
                elif name.startswith("compression"):
                    self.set_compressed_image(path_dir + file)
        return True
    
    def set_roi(self, x, y, w, h, xy_type="absolute", dim_type="absolute"):
        self.set_arm_roi(x, y, w, h, xy_type, dim_type)
        self.set_uncompressed_roi(x, y, w, h, xy_type, dim_type)
        self.set_compressed_roi(x, y, w, h, xy_type, dim_type)
        self.set_measure_roi(x, y, w, h, xy_type, dim_type)
        self.set_box_roi(x, y, w, h, xy_type, dim_type)
    
    def display_results(self):
        self.total = 5 + len(self.compress_obj)
        
        cv2.destroyWindow(self.name)
        cv2.namedWindow(self.name)
        self._display_update(self.pos)
        cv2.cv.CreateTrackbar(self.bar, self.name, 0, 
                              self.total-1, self._display_update)
        
        while True:
            k = cv2.waitKey()
            self.pos = cv2.getTrackbarPos(self.bar, self.name)
            if k == 27 or k == ord('q'): # ESC
                break
            if k == 9: # tab
                self._display_update(0)
                cv2.waitKey(500)
            elif k == ord('a'): # left arrow
                self.pos = (self.pos - 1) % self.total
                cv2.setTrackbarPos(self.bar, self.name, self.pos)
            elif k == ord('d'): # right arrow
                self.pos = (self.pos + 1) % self.total
                cv2.setTrackbarPos(self.bar, self.name, self.pos)
            elif k == ord('s'):
                self.seg = (self.seg + 1) % 3
            elif k == ord('r'):
                self.rect = (self.rect + 1) % 3
            else:
                continue
            self._display_update(self.pos)
            
        cv2.destroyAllWindows()
        return
                 
    def _display_update(self, index):
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
    parser = argparse.ArgumentParser(description="Process Baxter experiment images.")  
    parser.add_argument("-v", "--view", action="store_true", 
                        help="display results in window")
    parser.add_argument("-e", "--export", nargs=1, metavar="DIR",
                        help="export results to file directory")
    parser.add_argument("-d", "--dir", nargs=1, 
                        help="load directory path of images to add")
    parser.add_argument("-b", "--bg", nargs=1, metavar="FILE", 
                        help="add background image")
    parser.add_argument("-m", "--measure", nargs=1, metavar="FILE",
                        help="add measure reference image")
    parser.add_argument("-m-d", "--measure-dim", nargs=2, type=int, 
                        metavar=("WIDTH", "HEIGHT"),
                        help="specify measure reference dimensions")
    parser.add_argument("-x", "--box", nargs=1, metavar="FILE",
                        help="add box reference image")
    parser.add_argument("-a", "--arm", nargs=1, metavar="FILE",
                        help="add manipulating arm image")
    parser.add_argument("-a-r", "--arm-color-range", nargs=3, type=int, 
                        metavar=("HUE", "SATURATION", "VALUE"),
                        help="specify arm color tolerance (in HSV space)")
    parser.add_argument("-o", "--obj", nargs=1, metavar="FILE",
                         help="add uncompressed object image")
    parser.add_argument("-c", "--compression", nargs='+', metavar="FILE",
                        help="add compressed object image(s)")
    args = parser.parse_args()
    
    baxter = BaxterExperiment()
    if args.dir:
        print "Importing files from", args.dir[0], "...",
        baxter.import_images(args.dir[0])
        print "done."
    if args.bg:
        print "Setting background image to", args.bgv, "...",
        baxter.bg_path = args.bg[0]
        print "done."
    if args.measure:
        f = args.measure[0]
        if args.measure_dim:
            w = args.measure[1]
            h = args.measure[2]
            print ("Setting measurement reference image to " + str(f) 
                   + "with known" + w + "x" + h + " mm dimensions ..."),
            baxter.set_measure_image(f, w, h)
        else:
            print ("Setting measurement reference image to " 
                   + str(f) + "with DEFAULT dimensions ..."),
            baxter.set_measure_image(f)
        print "done."
    if args.box:
        print "Setting box reference image to", args.box[0], "...",
        baxter.set_box_image(args.box[0])
        print "done."
    if args.arm:
        f = args.arm[0]
        if args.arm_color_range:
            h = args.arm_color_range[0]
            s = args.arm_color_range[1]
            v = args.arm_color_range[2]
            print ("Setting arm image to " + str(f) + "with color ranges h="
                   + str(h) + ", s=" + str(s) + ", v=" + str(v) + " ..."),
            baxter.set_arm_image(f, h, s, v)
        else:
            print "Setting arm image to ", f, "with DEFAULT color ranges...",
            baxter.set_arm_image(f)
        print "done."
    if args.obj:
        print "Setting uncompressed object image to", args.obj[0], "...",
        baxter.set_uncompressed_image(args.obj[0])
        print "done."
    if args.compression:
        print "Setting compressed object image(s) to", args.compression[0], "...",
        for f in args.compression:
            baxter.set_uncompressed_image(args.compression)
        print "done."
    if baxter.bg_path:
        print "Baxter experiment successfully loaded. Have some stats:"    
        print "Color range:", baxter._color_low, baxter._color_high  
        print "Millimeters / Pixel:", baxter.get_mm_per_px()
        print "Measure object size (px):", baxter.get_measure_size()
        print "Box object size (px):", baxter.get_box_size()
        print "Object object size (px):", baxter.get_uncompressed_size()
        print "Compressed object size (px):", baxter.get_compressed_size()
        
    if args.export:
        print "Exporting results to", args.export[0], "..."
        if not baxter.export_results(args.export[0]):
            print "Nothing written. Are you sure that's a directory?"
    if args.view:
        print "Opening results window ...",
        baxter.display_results()
        print "closed."
    print "Finished executing. Goodbye."
    return

if __name__ == "__main__":
    main() 
