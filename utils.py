'''
FILE          : utils.py
AUTHOR        : LiuShuwei
CREAT DATE    : 2020-09-29
MODIFIED DATE : 2020-10-18
'''
#%%
import numpy as np
import os, time
import csv
from skimage import transform, io

#==根据文件名读图片==#
def read_image(filename):
    x = io.imread(filename)
    # x = np.array(img.imread(filename))
    # x = transform.resize(x,(107,107)) #(h,w)新图像尺寸
    return x

#==获取目录下指定后缀的文件名==#
def getFiles(dir, suffix): # 目录，文件后缀 
    res = []
    for root, directory, files in os.walk(dir):    # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename) # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename))
    return res

def read_csv(filename, dtype=str):
    with open(filename, 'r', encoding="utf-8") as f:
        rows = csv.reader(f)
        rows = [[dtype(value) for value in row] for row in rows]
    return np.array(rows)


def validGrdth(grdth, imageShape):
    '''(x1, y1, x2, y2, ...) => (cx, cy, w, h)'''
    x  = grdth[0::2] # 横坐标，列
    y  = grdth[1::2] # 纵坐标，行
    cx = np.average(x)
    cy = np.average(y)
    w  = min(np.max(x)-np.min(x), (imageShape[1]-cx)*2-1)
    h  = min(np.max(y)-np.min(y), (imageShape[0]-cy)*2-1)
    return np.array([cx, cy, w, h])


def shuffle(images, labels):
    assert images.shape[0] == labels.shape[0]
    idx = np.random.permutation(np.arange(0, images.shape[0])).tolist()
    return images[idx], labels[idx]


class Timer:
    time = time.time()
    @classmethod
    def tic(self):
        self.time = time.time()
    @classmethod
    def toc(self):
        print(time.time() - self.time)
        self.time = time.time()


class Iter:
    '''return shuffled input array'''
    def __init__(self, array, repeat=True, stride=8):
        self.data    = np.random.permutation(array)
        self.pointer = 0
        self.dataLen = len(self.data)
        self.repeat  = repeat
        self.stride  = stride

    def __iter__(self):
        return self
        
    def __next__(self): 
        if self.pointer + self.stride > self.dataLen:
            if self.repeat is False:
                raise StopIteration
            else:
                res = self.data[self.pointer:self.dataLen]
                self.data    = np.random.permutation(self.data)
                self.pointer = self.stride-(self.dataLen-self.pointer)
                res = np.append(res, self.data[:self.pointer], axis=0)
                
        else:
            res = self.data[self.pointer:self.pointer+self.stride]
            self.pointer = self.pointer + self.stride
        
        assert len(res) == self.stride, 'res length error'
        return res
