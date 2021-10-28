import numpy as np
import cv2
from measure import measureCommand

import sys

from tic import Tic
sys.setrecursionlimit(100000)

class Region:
    
    def __init__(self, id, pixs):
        # print(f'region {id} init')
        self.id = id
        self.pixs = pixs

    def __str__(self):
        return 'region {0}, area:{1:4d}, bounding box:{2}'.format(self.id, self.area, self.bbox)

    @property
    def pixs(self):
        return self._pixs

    @pixs.setter
    def pixs(self, pixs):
        if isinstance(pixs, list):
            if len(pixs) > 0:
                self._pixs = pixs
            else:
                raise ValueError('Expected None Empty list for argument pixs')
        else:
            raise TypeError('Expected list() for argument pixs')

    @property
    def id(self):
        return self._id 

    @id.setter
    def id(self, id):
        if isinstance(id, int):
            self._id = id
        else: 
            raise TypeError('Expected int for argument id') 

    @property
    def area(self):
        return len(self._pixs)

    @property
    def bbox(self):
        xs = [pix[0] for pix in self.pixs]
        ys = [pix[1] for pix in self.pixs]
        return (min(xs), min(ys), max(xs), max(ys))


def regionGrowing(img, i, j):
    img[i, j] = 0
    res = [(i,j)]
    for m,n in [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]:
        if m>=0 and n>=0 and m<img.shape[0] and n<img.shape[1]:
            if img[m, n] == 255:
                res.extend(regionGrowing(img, m, n))
    return res


@measureCommand
def instanceDetect(img, stride):
    img = img.copy()
    if len(img.shape) == 3:
        img = img[:,:,0]
    id = 0
    res = list()
    for i in range(img.shape[0]//stride):
        for j in range(img.shape[1]//stride):
            m, n = i*stride, j*stride
            if img[m:m+stride, n:n+stride].sum() >= 255:
                # print(m,n)
                for x in range(stride):
                    for y in range(stride):
                        if img[m+x, n+y] == 255:
                            region = Region(id, regionGrowing(img, m+x, n+y))
                            res.append(region)
                            for pix in region.pixs:
                                img[pix] = 0
                            id += 1
    return res
    

@measureCommand
def drawRes(img, res):
    img = img.copy()
    colors = [(255,90,90), (90,255,90), (90,90,255), (255,255,90), (255,90,255), (90,255,255)]
    for region in res:
        bbox = region.bbox
        rec = (bbox[1], bbox[0], bbox[3]-bbox[1]+1, bbox[2]-bbox[0]+1)
        cv2.putText(img, 
            text='{0},{1}'.format(region.id, bbox), 
            org=(rec[0], rec[1]-2), 
            fontFace=cv2.FONT_HERSHEY_COMPLEX, 
            fontScale=.25, 
            color=colors[region.id%len(colors)], 
            thickness=1
        )
        for pix in region.pixs:
            img[pix] = colors[region.id%len(colors)]
        # cv2.rectangle(img, rec=rec, color=colors[region.id%len(colors)], thickness=1)

    return img

if __name__ == '__main__':

    import time

    import matplotlib.pyplot as plt

    img = np.zeros(shape=(1000, 2000, 3), dtype=np.uint8)
    o1  = np.zeros(shape=(50, 50, 3), dtype=np.uint8) + 255

    for x,y in [(10,10), (300,50), (100,800), (200,1000), (900, 1400), (500, 1300)]:
        img[x:x+o1.shape[0], y:y+o1.shape[1]] = o1

    # cv2.namedWindow('img', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    # cv2.imshow('img', img)
    
    # ts = list()
    # for stride in range(1, 100):
    #     t = time.time()  
    #     res = instanceDetect(img, stride)
    #     ts.append(time.time() - t)
    # plt.plot(10*np.log10(np.array(ts)/min(ts)))
    # plt.show()

    res = instanceDetect(img, 30)

    img = drawRes(img, res)

    cv2.namedWindow('res', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow('res', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
