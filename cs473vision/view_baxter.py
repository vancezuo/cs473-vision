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
    '''
    A BaxterExperiment is a BaxterObject with methods to facilitate the 
    use a BaxterObject's functions. It contains methods for importing images
    and exporting results en masse, as well as displaying the result images
    (along with segments and bounding rectangles) in a window.
    
    A notable method is display_results(), which brings up result images of
    the segmentation algorithm in a window. On Windows, it can also accept
    keyboard input:
        
        - Pressing ESC or 'q' closes the window.
        - Pressing 'a' moves the slider a tick left, and 'A' 5 ticks.
        - Pressing 'd' moves the slider a tick right, and 'D' 5 ticks.
        - Pressing 's' toggles what segment of the image to display.
        - Pressing 'r' toggles what bounding rectangle to display.
        - Pressing TAB temporarily displays the background image, allowing
          for quick comparison between the background and current image.
    
    '''
    
    def __init__(self, bg_file=None):
        '''
        Initiates BaxterExperiment, with (optionally) a user-specified 
        background image.
        
        Args:
            bg_path: file path to background image.
        '''
        
        super(BaxterExperiment, self).__init__(bg_file)
        self._name = "BaxterObject"
        self._bar = "Image"
        
        self._pos = 0
        self._total = 1
        self._seg = 0 # 0 = none, 1 = region, 2 = object
        self._rect = 2 # 0 = none, 1 = upright, 2 = min area
        return
    
    def export_results(self, output_dir, segment=True, roi=False, table=True):
        '''
        Initiates BaxterExperiment, with (optionally) a user-specified 
        background image.
        
        Args:
            output_dir: directory path to write output images to.
        Returns:
            True if the output directory is valid; false otherwise.
        '''
        
        if not os.path.isdir(output_dir):
            return False
        if not output_dir.endswith("/"):
            output_dir += "/"
        if segment:
            self.export_measure_segment(output_dir+"reference-_seg.png")
            self.export_arm_segment(output_dir+"arm-_seg.png")
            self.export_uncompressed_segment(output_dir+"object-_seg.png")
            self.export_compress_segment(output_dir+"compression-_seg.png")      
        if roi:
            self.export_measure_roi_segment(output_dir+"reference-roi.png")
            self.export_arm_roi_segment(output_dir+"arm-roi.png")
            self.export_uncompressed_roi_segment(output_dir+"object-roi.png")
            self.export_compress_roi_segment(output_dir+"compression-roi.png")
        if table:
            self.export_sizes(output_dir + "sizes.csv")
        return True

    def print_results(self):
        '''
        Prints out various results of the BaxterExperiment's image processing:
        the arm color range, the millimeter to pixel conversion factor, the
        measurement object reference pixel size, the box object size,
        the uncompressed object pixel size, and the (smallest area) compressed 
        object pixel size.
        '''
        
        print "Color range:", self._color_low, self._color_high  
        print "Millimeters / Pixel:", self.get_mm_per_px()
        print "Measure object size (px):", self.get_measure_size()
        print "Box object size (px):", self.get_box_size()
        print "Object object size (px):", self.get_uncompressed_size()
        print "Compressed object size (px):", self.get_compressed_size()
        return
    
    def import_images(self, path_dir): # Caution: very specific
        '''
        Loads images from a directory into the BaxterExperiment. The specific 
        naming convention for the images is as follows: the background image is 
        "background"/"bg", the reference object image is "reference"/"ref", 
        the arm image is "arm", the box image is "box", the uncompressed 
        object image is "object"/"obj", and the compressed object images 
        start with "compression". Images that are not named this way
        are ignored.
        
        The method only reads PNG or JPG image files. Also note that the 
        compression images are added in alphabetical order.
        
        Args:
            path_dir: directory path of the images to load.
        Returns:
            True if the input directory is valid; false otherwise.
        '''
        
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
        for file in sorted(os.listdir(path_dir)):
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
        '''
        Sets the rectangular region of interest for all images that are loaded
        into BaxterExperiment.
        
        Note that there is no check for validity.
        
        Args:
            x: integer x-value of top-left point of the ROI rectangle.
            y: integer y-value of top-left point of the ROI rectangle.
            w: integer width (x-dimension) of the ROI rectangle.
            h: integer height (y-dimension) of the ROI rectangle.
            xy_type: 'absolute' if (x,y) are to be interpreted as absolute
                     pixel values; 'relative' if (x,y) are percentages of 
                     overall image from which to determine the top-left corner
                     pixel.
            dim_type: 'absolute' if (w,h) are to be interpreted as absolute
                      pixel dimensions; 'relative' if (x,y) are percentages of 
                      overall image from which to determine pixel dimensions.
        '''
        
        self.set_arm_roi(x, y, w, h, xy_type, dim_type)
        self.set_uncompressed_roi(x, y, w, h, xy_type, dim_type)
        self.set_compressed_roi(x, y, w, h, xy_type, dim_type)
        self.set_measure_roi(x, y, w, h, xy_type, dim_type)
        self.set_box_roi(x, y, w, h, xy_type, dim_type)
        return
    
    def display_results(self):
        '''
        Opens a window and displays the results of the BaxterExperiment's
        segmentation of its object images. The window contains a slider
        which the user can move to toggle between different image results.
        It also accepts keyboard input:
        
            - Pressing ESC or 'q' closes the window.
            - Pressing 'a' moves the slider a tick left, and 'A' 5 ticks.
            - Pressing 'd' moves the slider a tick right, and 'D' 5 ticks.
            - Pressing 's' toggles what segment of the image to display.
            - Pressing 'r' toggles what bounding rectangle to display.
            - Pressing TAB temporarily displays the background image, allowing
              for quick comparison between the background and current image.
            
        This method does not terminate until the user closes the window. Note 
        also that the keyboard functions have been tested to only completely 
        work on Windows.
        '''
        
        self._total = 5 + len(self.compress_obj)
        
        #cv2.namedWindow(self._name)
        self._display_update(self._pos)
        cv2.cv.CreateTrackbar(self._bar, self._name, 0, 
                              self._total-1, self._display_update)
        
        while True:
            k = cv2.waitKey()
            self._pos = cv2.getTrackbarPos(self._bar, self._name)
            if k == 27 or k == ord('q') or k == -1: # ESC or no key press
                break
            if k == 9: # tab
                self._display_update(0)
                cv2.waitKey(500)
            elif k == ord('a'): # left arrow
                self._pos = (self._pos - 1) % self._total
                cv2.setTrackbarPos(self._bar, self._name, self._pos)
            elif k == ord('d'): # right arrow
                self._pos = (self._pos + 1) % self._total
                cv2.setTrackbarPos(self._bar, self._name, self._pos)
            elif k == ord('A'): # left arrow * 5
                self._pos = (self._pos - 5) % self._total
                cv2.setTrackbarPos(self._bar, self._name, self._pos)
            elif k == ord('D'): # right arrow * 5
                self._pos = (self._pos + 5) % self._total
                cv2.setTrackbarPos(self._bar, self._name, self._pos)
            elif k == ord('s'):
                self._seg = (self._seg + 1) % 3
            elif k == ord('r'):
                self._rect = (self._rect + 1) % 3
            else:
                continue
            self._display_update(self._pos)
        
        cv2.waitKey(-1) # for Linux
        cv2.destroyWindow(self._name)
        cv2.imshow(self._name, np.array([0])) # for Linux
        return
                 
    def _display_update(self, index):
        bg_img = cv2.imread(self.bg_path)
        if index == 0:
            # Apply the same blurring filter as in object_detect.py
            cv2.imshow(self._name, cv2.bilateralFilter(bg_img, 5, 100, 100))
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
            cv2.imshow(self._name, black_img)
            return
             
        if self._seg == 2:
            obj_mask = obj.get_object_mask()
            img = cv2.bitwise_and(obj.fg_img, obj.fg_img, mask=obj_mask)
        elif self._seg == 1:
            region_mask = cv2.bitwise_and(obj.rect_mask, obj.color_mask)
            img = cv2.bitwise_and(obj.fg_img, obj.fg_img, mask=region_mask)
        else:
            img = obj.fg_img.copy()
            
        if self._rect >= 1:
            points = np.int0(obj.get_object_rectangle_points(self._rect == 2))
            cv2.drawContours(img, [points], 0, (255,255,255), 2)
            
        cv2.imshow(self._name, img)
        return

# Test script for BaxterExperiment
def main():
    parser = argparse.ArgumentParser(description="Process Baxter experiment images.")  
    parser.add_argument("-v", "--view", action="store_true", 
                        help="display results in window")
    parser.add_argument("-e", "--export", nargs=1, metavar="DIR",
                        help="export results to file directory")
    parser.add_argument("-i", "--import", nargs=1, metavar="DIR", dest="dir",
                        help="load directory path of images to add")
    parser.add_argument("-ie", nargs=1, metavar="DIR",
                        help="load directory path of images and export to same")
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
    elif args.ie:
        print "Importing files from", args.ie[0], "...",
        baxter.import_images(args.ie[0])
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
        baxter.print_results()
        
    if args.export:
        print "Exporting results to", args.export[0], "...",
        if baxter.export_results(args.export[0]):
            print "done."
        else:
            print "nothing written. Are you sure that's a directory?"
    elif args.ie:
        print "Exporting results to", args.ie[0], "...",
        if baxter.export_results(args.ie[0]):
            print "done."
        else:
            print "nothing written. Are you sure that's a directory?"        
            
    if args.view:
        print "Opening results window ...",
        baxter.display_results()
        print "closed."
    print "Finished executing. Goodbye."
    return

if __name__ == "__main__":
    main() 
