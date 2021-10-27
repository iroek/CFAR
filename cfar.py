# import
import sys
import cv2
import numpy as np
import random
import time
import os
from tic import Tic

# configs
GUARD_CELLS = 5
BG_CELLS = 10
ALPHA = 1
CFAR_UNITS = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)
HALF_CFAR_UNITS = int(CFAR_UNITS/2) + 1

#path
OUTPUT_IMG_DIR = "./test_out/"
root='./test/'

#2D-CA-CFAR
def cfar(img_path):
    inputImg = cv2.imread(img_path, 0)
    out_name=os.path.basename(img_path).split('.')[0]
    estimateImg = np.zeros((inputImg.shape[0], inputImg.shape[1], 3), np.uint8)
    # search
    for i in range(inputImg.shape[0] - CFAR_UNITS):
        print(i)
        center_cell_x = i + BG_CELLS + GUARD_CELLS
        Tic.tic()
        for j in range(inputImg.shape[1] - CFAR_UNITS):
            center_cell_y = j  + BG_CELLS + GUARD_CELLS
            average = 0
            for k in range(CFAR_UNITS):
                for l in range(CFAR_UNITS):
                    if (k >= BG_CELLS) and (k < (CFAR_UNITS - BG_CELLS)) and (l >= BG_CELLS) and (l < (CFAR_UNITS - BG_CELLS)):
                        continue
                    average += inputImg[i + k, j + l]
            average /= (CFAR_UNITS * CFAR_UNITS) - ( ((GUARD_CELLS * 2) + 1) * ((GUARD_CELLS * 2) + 1) )
        
            
            if inputImg[center_cell_x, center_cell_y] > (average * ALPHA):
                estimateImg[center_cell_x, center_cell_y] = (0,0,255)
        Tic.toc()
    # output
    tmpName = OUTPUT_IMG_DIR + f"{out_name}_{GUARD_CELLS}_{BG_CELLS}_{ALPHA}.png"
    cv2.imwrite(tmpName, estimateImg)

if __name__=='__main__':
    imgs=os.listdir(root)
    for img in imgs:
        print(img)
        img_path=root+img
        # print(img_path)
        cfar(img_path)