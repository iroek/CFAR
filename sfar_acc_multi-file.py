# import
import sys
import cv2
import numpy as np
import random
import time
import os
from tic import Tic
from multiprocessing import Pool
from sfar_once import *

# configs
GUARD_CELLS = 10
BG_CELLS    = 5
ALPHA       = 2

#path
OUTPUT_IMG_DIR = "./test_out/"
root='./test/'


#2D-CA-CFAR
def cfar(arg):
    print(arg)
    img_path = arg.get('img_path')
    gc       = arg.get('gc')
    bc       = arg.get('bc')
    al       = arg.get('al')
    inputImg    = cv2.imread(img_path, 0).astype(float)
    out_name    = os.path.basename(img_path).split('.')[0]
    estimateImg = np.zeros((inputImg.shape[0], inputImg.shape[1]), np.uint8)
    CFAR_UNITS      = 1 + (gc * 2) + (bc * 2)
    HALF_CFAR_UNITS = gc + bc
    arg['inputImg']        = inputImg
    arg['CFAR_UNITS']      = CFAR_UNITS
    arg['HALF_CFAR_UNITS'] = HALF_CFAR_UNITS

    # search
    res = list()
    for i in range(inputImg.shape[0]-CFAR_UNITS):
        for j in range(inputImg.shape[1]-CFAR_UNITS):
            res.append(sfar_one_improved(dict_merge(arg, {'index':(i,j)}) ))
    for p in res:
        estimateImg[p[0], p[1]] = p[2]

    # output
    tmpName = OUTPUT_IMG_DIR + f"{out_name}_{gc}_{bc}_{al}.png"
    cv2.imwrite(tmpName, estimateImg)
    

if __name__=='__main__':
    imgs=os.listdir(root)
    
    for img in imgs:
        img_path=root+img
        Tic.tic()
        with Pool(12) as pool:
            pool.map(cfar, [{'img_path':img_path, 'gc':GUARD_CELLS, 'bc':BG_CELLS, 'al':ALPHA} for ALPHA in [1.6, 1.8, 2] for GUARD_CELLS in range(15, 25) for BG_CELLS in range(5,15)]) 
        Tic.toc()

    # with Pool(16) as pool:
    #     pool.map(cfar, [{'img_path':root+img, 'gc':GUARD_CELLS, 'bc':BG_CELLS, 'al':ALPHA} for img in imgs]) 