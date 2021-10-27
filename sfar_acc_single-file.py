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
GUARD_CELLS = 50
BG_CELLS    = 50
ALPHA       = 1.8

#path
OUTPUT_IMG_DIR = "./test_out/"
root='./test/'


if __name__=='__main__':

    #2D-CA-CFAR
    def cfar(arg):
        # print(arg)
        img_path    = arg.get('img_path')
        gc          = arg.get('gc')
        bc          = arg.get('bc')
        al          = arg.get('al')
        inputImg    = cv2.imread(img_path, 0).astype(float)
        out_name    = os.path.basename(img_path).split('.')[0]
        estimateImg = np.zeros((inputImg.shape[0], inputImg.shape[1]), np.uint8)
        CFAR_UNITS  = 1 + (gc * 2) + (bc * 2)
        HALF_CFAR_UNITS = gc + bc
        arg['inputImg'] = inputImg
        arg['CFAR_UNITS']      = CFAR_UNITS
        arg['HALF_CFAR_UNITS'] = HALF_CFAR_UNITS
         
        # search
        with Pool(16) as p:
            res = p.map(sfar_one_improved, [dict_merge(arg, {'index':(i,j)}) for i in range(inputImg.shape[0]-CFAR_UNITS) for j in range(inputImg.shape[1]-CFAR_UNITS)])
        for p in res:
            estimateImg[p[0], p[1]] = p[2]

        # output
        tmpName = OUTPUT_IMG_DIR + f"{out_name}_{gc}_{bc}_{al}.png"
        cv2.imwrite(tmpName, estimateImg)

    imgs=os.listdir(root)
    for img in imgs:
        img_path=root+img
        Tic.tic()
        cfar({'img_path':img_path, 'gc':GUARD_CELLS, 'bc':BG_CELLS, 'al':ALPHA})
        Tic.toc()
