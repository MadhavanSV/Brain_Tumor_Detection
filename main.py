import numpy as np
import argparse
import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join
#from Mask_Detection.mask_detector import maskdetection
from Mask_Detection.photo import maskdetection
#construct the arguement parser and parse the arguements.
ap = argparse.ArgumentParser()
ap.add_argument("-p","--path", required = True,help = "Path of the input video")
args = vars(ap.parse_args())
maskdetection(args["path"])


