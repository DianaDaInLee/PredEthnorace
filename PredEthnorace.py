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

def img_search(fullname, maxnum, imgfolder):
    url = "https://duckduckgo.com/"
    keywords = fullname.lower()
    subfolder = keywords.replace(" ", "_")

    time.sleep(random.choice([1, 2, 3, 4, 5]))
    try:
        data = ddg_images(keywords)

    except:
        try:
            time.sleep(random.choice([1, 2, 3, 4, 5]))
            data = ddg_images(keywords)
        except: 
            print("ERROR: " + keywords + "not searchable." )

    else:
        i = 0 #image name counter
        j = 0 #total success counter
        if not os.path.exists(imgfolder):
            os.mkdir(imgfolder)

        if os.path.exists(os.path.join(imgfolder, subfolder)):
            print(subfolder + 'already exist and was replaced')
            shutil.rmtree(os.path.join(imgfolder, subfolder))
        
        if not os.path.exists(os.path.join(imgfolder, subfolder)):
            os.mkdir(os.path.join(imgfolder, subfolder))
        
        while j < maxnum:
            print(str(j) + " total successful images collected. Trying image #" + str(i+1) + "\n") 
            fn        = subfolder + "_" + str(i) + ".jpg"
            file_path = os.path.join(imgfolder, subfolder, fn)
            try:
                urllib.request.urlretrieve(data[i]['thumbnail'], file_path)

            except:
                try:
                    urllib.request.urlretrieve(data[i]['image'], file_path)
                except:
                    i = i + 1
                    continue

            if (~os.path.isfile(file_path)):   
                time.sleep(random.choice([4,5]))

            if (os.path.isfile(file_path)):
                # Check how many faces there are
                faces = MTCNN().detect_faces(pyplot.imread(file_path))
                
                if ((len(faces)==1) and (j == 0)):
                    i = i + 1
                    j = j + 1
                elif ((len(faces)==1) and (j > 0)):
                    print('# Check if face already exists')
                    img_list = glob.glob(imgfolder + "/" + subfolder + "/*jpg")
                    img_list = [i for i in img_list if i != file_path]
                    dup_found = 0
                    for filename in img_list:
                      pairs_chk = DeepFace.verify(img1_path = filename, 
                                                  img2_path = file_path,
                                                  enforce_detection = False, 
                                                  model_name = 'OpenFace', 
                                                  distance_metric = 'euclidean_l2')
                      if pairs_chk['verified']==True:
                          dup_found = 1
                          break
                    if (dup_found==0):
                        print('All good with dupes, move on!')
                        i = i + 1
                        j = j + 1
                    else:
                        print('Duplicate face: ' + filename)
                        os.remove(file_path)
                        i = i + 1
                else:
                    print('Too many faces!')
                    os.remove(file_path)
                    i = i + 1
                    
def img_pred(imgfolder, subfolder = "", det = "opencv", out_csv = True):  
    subfolder = subfolder.replace(" ", "_")
    if (subfolder == ""):
        img_list = glob.glob(imgfolder + "/*/" + "*jpg")
    else:
        img_list = glob.glob(imgfolder + "/" + subfolder + "/" + "*jpg")

    if len(img_list)==0:
        print('Error: No images exist in folder' + imgfolder + "/" + subfolder)

    else:
        all = []
        for im in img_list:
            prd = DeepFace.analyze(im, ['race'], detector_backend = det, enforce_detection= False)
            df  = pd.DataFrame(prd).T
            res = pd.concat([df.iloc[0,:].apply(pd.Series), df.iloc[1,:]], axis = 1)
            res['fn'] = im
            res['detector'] = det
            all.append(res)

        all = pd.concat(all)
        if (out_csv == True):
            all.to_csv(imgfolder + "/img_pred.csv", index = False)

        return(all)