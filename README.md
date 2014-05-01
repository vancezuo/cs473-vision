cs473-vision
============

This repository contains object image processing modules for Samuel Goldstein, Thomas Weng, and Vance Zuo's semester project for Yale's CPSC 473 - Intelligent Robotics Lab course.

Dependencies / Requirements
---------------------------
* Python 2.7
* OpenCV 2.4

Installation
------------
The modules require the OpenCV and NumPy Python packages to be installed in the system they are run on. See http://opencv.org to download OpenCV for your operating system, if it isn't already present. This project was developed specifically with OpenCV 2.4.8, so we can only guarantee it will work correctly with that version. See http://www.numpy.org/ for NumPy.

Usage
-----
This package was designed to provide submodules for other parts of the project, but obj\_detect.py and view\_baxter.py can be run as standalone command-line python applications. 

### obj_detect.py

This simple module, or more accurately its SegmentedObject class, is meant to serve as a building block for more complex object image processing scenarios, either as a component in a larger class, as was done in the obj_baxter.py module, or by extending the class. It can segment an object from a foreground image based on a background image, using a background subtraction technique. 

The command-line application's two required arguments are background and foreground image file paths, respectively, with optional arguments for specifying a color range to ignore, a rectangular region of interest, or a specific segmentation method (currently, "simple", "mog", and "mog2" are the implemented). For more details, refer to the help text (-h option):

    usage: obj_detect.py [-h]
                         [-c HUE_LOW SAT_LOW VAL_LOW HUE_HIGH SAT_HIGH VAL_HIGH]
                         [-r X Y WIDTH HEIGHT] [-m METHOD]
                         background foreground
    
    Segment object from background.
    
    positional arguments:
      background            path to background image
      foreground            path to foreground image
    
    optional arguments:
      -h, --help            show this help message and exit
      -c HUE_LOW SAT_LOW VAL_LOW HUE_HIGH SAT_HIGH VAL_HIGH, --color HUE_LOW SAT_LOW VAL_LOW HUE_HIGH SAT_HIGH VAL_HIGH
                            specify ignored color
      -r X Y WIDTH HEIGHT, --rectangle X Y WIDTH HEIGHT
                            specify rectangle region of interest
      -m METHOD, --method METHOD
                            specify segmentation method
                            
### obj_baxter.py

This module's BaxterObject class represents the experimental scenario in the project, which contains multiple objects. It store images of the target object in uncompressed and compressed forms, a reference object of known dimensions (for converting pixels per millimeter), a box object that we are trying to fit the target object in, and the robot arm (whose color will ignored in the compression images). These images are segmented based on a common background image as instances of the SegementedObject class. 

Although this module was designed with the project's experimental setup in mind, it can be easily extended to accommodate similar but slightly different setups. The class is also modularized so that it can function without all of the specified object images--for example, in the project's experiments, the box object was omitted due to a change in the project's goals. Its calculations relating to the box were meaningless, of course, but anything that didn't directly involve the box could calculated and output.

No application version of this module was created, as view_baxter.py (see below) is similar and already has a command-line interface.
                            
### view_baxter.py

This module's BaxterExperiment class builds upon BaxterObject's functionality for importing images and exporting results en masse, as well as visually displaying the result images (along with segments and bounding rectangles) in a window. Because the methods of this class do not extend naturally to child classes of BaxterObject (as they wouldn't accommodate any new instance fields in the child classes), they were separated and consolidated into their own module. For a summary of its command-line usage, refer to the help text (-h option):

    usage: view_baxter.py [-h] [-v] [-e DIR] [-i DIR] [-ie DIR] [-b FILE]
                          [-m FILE] [-m-d WIDTH HEIGHT] [-x FILE] [-a FILE]
                          [-a-r HUE SATURATION VALUE] [-o FILE]
                          [-c FILE [FILE ...]]
    
    Process Baxter experiment images.
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --view            display results in window
      -e DIR, --export DIR  export results to file directory
      -i DIR, --import DIR  import images from file directory
      -ie DIR               load images from directory path and export to same directory
      -b FILE, --bg FILE    add background image
      -m FILE, --measure FILE
                            add measure reference image
      -m-d WIDTH HEIGHT, --measure-dim WIDTH HEIGHT
                            specify measure reference dimensions
      -x FILE, --box FILE   add box reference image
      -a FILE, --arm FILE   add manipulating arm image
      -a-r HUE SATURATION VALUE, --arm-color-range HUE SATURATION VALUE
                            specify arm color tolerance (in HSV space)
      -o FILE, --obj FILE   add uncompressed object image
      -c FILE [FILE ...], --compression FILE [FILE ...]
                            add compressed object image(s)
                            
For most purposes, the -v, -e, -i, and -ie options will be sufficient. Take note, however, the -i and -ie options require either *.jpg or *.png images in the specified directory that follow strict naming conventions:

* Name of "background" or "bg" denotes the background image.
* Name of "reference" or "ref" denotes the reference object image.
* Name of "arm" denotes the robot arm image.
* Name of "box" denotes the box image.
* Name of "object" or "obj" denotes the uncompressed target object image.
* Names _starting with_ "compression" denote compressed object images. The order they are loaded is alphabetical.

The -v option loads a window displaying segmentation results for all of the loaded images, navigable by a slider. This can be useful for quickly toggling through the results of the segmentation algorithm. On Window, it also accepts keyboard input:
        
* Pressing ESC or 'q' closes the window.
* Pressing 'a' moves the slider a tick left, and 'A' (Shift+'a') 5 ticks left.
* Pressing 'd' moves the slider a tick right, and 'D' (Shift+'d') 5 ticks right.
* Pressing 's' toggles what segment of the image to display.
* Pressing 'r' toggles what bounding rectangle to display.
* Pressing TAB temporarily displays the background image, allowing
  for quick comparison between the background and current image.

Other operating systems may also be able to register keyboard input, but it is not guaranteed.
