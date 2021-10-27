# import
import sys
import cv2
import numpy as np
import random
import time
import os
from tic import Tic
from multiprocessing import Pool
from cfar_once import *
from utils import getFiles, get_yaml_data, dict_merge


#2D-CA-CFAR
def cfar(arg):
    print(arg)
    img_path    = arg.get('img_path')
    gc          = arg.get('GUARD_CELLS')
    bc          = arg.get('BG_CELLS')
    al          = arg.get('ALPHA')
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
            res.append(cfar_one_improved(dict_merge(arg, {'index':(i,j)}) ))
    for p in res:
        estimateImg[p[0], p[1]] = p[2]

    # output
    if arg.get('saveResult'):
        tmpName = os.path.join(arg.get('OUTPUT_IMG_DIR'), f"{out_name}_{gc}_{bc}_{al}.png")
        cv2.imwrite(tmpName, estimateImg)
    
    return estimateImg


if __name__=='__main__':
    arg = get_yaml_data('config/arg.yaml')

    img_paths = getFiles(arg.get('root'), '.jpg')    
    for img_path in img_paths:
        Tic.tic()
        with Pool(10) as pool:
            pool.map(cfar, [dict_merge(arg, {'img_path':img_path, 'GUARD_CELLS':GUARD_CELLS, 'bc':GUARD_CELLS*RATIO, 'al':ALPHA}) for ALPHA in [1.2, 1.4, 1.6, 1.8, 2.] for GUARD_CELLS in [30, 40, 50, 60] for RATIO in [1,]]) 
        Tic.toc()
