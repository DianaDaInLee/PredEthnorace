#pip install duckduckgo_search
#pip install DeepFace
#pip install pyreadr
#pip install opencv-python

import sys
import os
import requests
import random
import re
import json
import time
import logging
import urllib
import scipy.misc
import cv2
import numpy as np
import glob
import socket
import pandas as pd
import pickle
import shutil
import tensorflow as tf
import keras
from duckduckgo_search import ddg_images
from deepface import DeepFace
from PIL import Image
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from socket import timeout
from matplotlib.pyplot import imread
socket.setdefaulttimeout(90)

def img_check(imgfolder, detector = 'opencv', distance = 'euclidean_l2', model = 'OpenFace'):
    img_list = glob.glob(imgfolder + "/*jpeg") + glob.glob(imgfolder + "/*jpg") + glob.glob(imgfolder + "/*png")
    
    # Check how many faces there are
    df_nface = pd.DataFrame(columns=['image', 'n_faces'])
    for img in img_list:
        faces = MTCNN().detect_faces(pyplot.imread(img))
        df_nface = df_nface.append({'image': img, 'n_faces': len(faces)}, ignore_index = True)

    # Check for duplicates
    unique_pairs = []
    for i in range(len(img_list)):
        for j in range(i + 1, len(img_list)):
            # Create a tuple representing a unique pair of elements
            pair = (img_list[i], img_list[j])
            unique_pairs.append(pair)

    df_dupes = pd.DataFrame(columns=['image1', 'image2', 'dupe', 'distance', 'model', 'detector'])
    for img in unique_pairs:
        pairs_chk = DeepFace.verify(img1_path = img[0], 
                                    img2_path = img[1],
                                    enforce_detection = False, 
                                    model_name = model, 
                                    distance_metric = distance,
                                    detector_backend= detector)
        df_dupes = df_dupes.append({'image1': img[0], 
                                    'image2': img[1],
                                    'dupe': pairs_chk['verified'],
                                    'distance': pairs_chk['distance'],
                                    'model': pairs_chk['model'],
                                    'detector': pairs_chk['detector_backend']}, ignore_index = True)
        
    return df_nface, df_dupes
                                    
            
                    
def img_pred(imgfolder, detector = "opencv", out_csv = True):
    img_list = glob.glob(imgfolder + "/*jpeg") + glob.glob(imgfolder + "/*jpg") + glob.glob(imgfolder + "/*png")
    
    if len(img_list)==0:
        print('Error: No images exist in folder' + imgfolder)

    else:
        all = []
        for im in img_list:
            prd = DeepFace.analyze(im, ['race'], detector_backend = detector, enforce_detection= False)
            df  = pd.DataFrame(prd).T
            res = pd.concat([df.iloc[0,:].apply(pd.Series), df.iloc[1,:]], axis = 1)
            res['fn'] = im
            res['detector'] = detector
            all.append(res)

        all = pd.concat(all)
        if (out_csv == True):
            all.to_csv(imgfolder + "/img_pred.csv", index = False)

        return all