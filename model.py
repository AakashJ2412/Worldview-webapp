import fastai.vision
import torch
from fastai.vision import *
from fastai.vision.learner import *
from fastai.vision.interpret import *
import torchvision.transforms as T
# from fastai.vision.all import *
# from fastai.vision.interpret import *
# from fastai.callbacks.hooks.all import *
# from fastai.vision.all import *
from pathlib import Path
from fastai.utils.mem import *
torch.backends.cudnn.benchmark=True
import os
import glob
import base64
import numpy as np

# os.system("load_ext cython")
# os.system("cython -a")

# import cython
# import numpy
# cimport numpy
import cv2 as cv

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

path = Path('./RTK/unzipped')
path.ls()


get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

codes = np.loadtxt(path/'codes.txt', dtype=str); codes

path_lbl = path/'labels'
path_img = path/'images'


# using saved model to predict
def load_model(model_path):

    size = (array([288, 352]))

    free = gpu_mem_get_free_no_cache()
    # the max size of bs depends on the available GPU RAM
    if free > 8200: bs=8
    else:           bs=4
    # print(f"using bs={bs}, have {free}MB of GPU RAM free")

    # hardcoding dataset
    src = (SegmentationItemList.from_folder(path_img)
        .split_by_fname_file('../valid.txt')
        .label_from_func(get_y_fn, classes=codes))

    data = (src.transform(get_transforms(), size=size, tfm_y=True)
            .databunch(bs=bs)
            .normalize(imagenet_stats))

    learn = unet_learner(data, models.resnet34)

    learn.load('stage-2-weights')

    return learn


# image to predict
def predict(model_path,img):

    learn = load_model(model_path)
    # img = open_image('iiith.jpg')
    # print(img)
    imgTensor = T.ToTensor()(img)
    img = Image(imgTensor)
    prediction = learn.predict(img)
    prediction[0].save('./results/prediction.png')

def colour(frame):
    width = 288
    height = 352
    for x in range(width):
        for y in range(height):
            b, g, r = frame[x, y]
            if (b, g, r) == (0,0,0): #background
              frame[x, y] = (0,0,0)
            elif (b, g, r) == (1,1,1): #roadAsphalt
              frame[x, y] = (85,85,255)
            elif (b, g, r) == (2,2,2): #roadPaved
              frame[x, y] = (85,170,127)
            elif (b, g, r) == (3,3,3): #roadUnpaved
              frame[x, y] = (255,170,127) 
            elif (b, g, r) == (4,4,4): #roadMarking
              frame[x, y] = (255,255,255) 
            elif (b, g, r) == (5,5,5): #speedBump
              frame[x, y] = (255,85,255)
            elif (b, g, r) == (6,6,6): #catsEye
              frame[x, y] = (255,255,127)          
            elif (b, g, r) == (7,7,7): #stormDrain
              frame[x, y] = (170,0,127) 
            elif (b, g, r) == (8,8,8): #manholeCover
              frame[x, y] = (0,255,255) 
            elif (b, g, r) == (9,9,9): #patchs
              frame[x, y] = (0,0,127) 
            elif (b, g, r) == (10,10,10): #waterPuddle
              frame[x, y] = (170,0,0)
            elif (b, g, r) == (11,11,11): #pothole
              frame[x, y] = (255,0,0)
            elif (b, g, r) == (12,12,12): #cracks
              frame[x, y] = (255,85,0)
 
    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    # return the colored image
    return frame

def runModel(image):
  model_path = './RTK/unzipped/images/models/stage-2-weights.pth'
  predict(model_path,image)
  outFrame = cv.imread('./results/prediction.png')
  outFrame = colour(outFrame)
  cv.imwrite('./results/colored.png',outFrame)
