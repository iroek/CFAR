#%%
from genericpath import isdir
import torch
from torch import nn
# from torch._C import uint8
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda, Compose
# import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

GUARD_CELLS = 80
BG_CELLS    = 80
ALPHA       = 1.8

#path
OUTPUT_IMG_DIR = "./test_out/"
root='./test/'
# root = r'D:\POLAR\USI2000\train'

# Define model
class Sarf(nn.Module):
    def __init__(self, GUARD_CELLS, BG_CELLS, ALPHA):
        super(Sarf, self).__init__()
        CFAR_UNITS  = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)
        HALF_CFAR_UNITS = GUARD_CELLS + BG_CELLS

        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=CFAR_UNITS, 
            stride=1, 
            padding=HALF_CFAR_UNITS
        )
        self.conv1.weight = torch.nn.Parameter(
            torch.ones_like(self.conv1.weight, dtype=torch.float32)
        )
        self.conv1.bias = torch.nn.Parameter(
            torch.zeros_like(self.conv1.bias, dtype=torch.float32)
        )    
            
        self.div1 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=1, 
            stride=1
        )
        self.div1.weight = torch.nn.Parameter(
            torch.div(
                torch.ones_like(self.div1.weight, dtype=torch.float32), 
                CFAR_UNITS ** 2 - (GUARD_CELLS * 2 + 1) **2
            )
        )
        self.div1.bias = torch.nn.Parameter(
            torch.zeros_like(self.div1.bias, dtype=torch.float32)
        ) 

        self.conv2 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(GUARD_CELLS * 2) + 1, 
            stride=1,
            padding=GUARD_CELLS
        )
        self.conv2.weight = torch.nn.Parameter(
            torch.ones_like(self.conv2.weight, dtype=torch.float32)
        )
        self.conv2.bias = torch.nn.Parameter(
            torch.zeros_like(self.conv2.bias, dtype=torch.float32)
        )   

        self.div2 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=1, 
            stride=1
        )
        self.div2.weight = torch.nn.Parameter(
            torch.div(
                torch.ones_like(self.div2.weight, dtype=torch.float32), 
                ( (GUARD_CELLS * 2 + 1) **2 ) * ALPHA
            )
        )
        self.div2.bias = torch.nn.Parameter(
            torch.zeros_like(self.div2.bias, dtype=torch.float32)
        ) 

    def forward(self, x):
        all = self.conv1(x)
        front = self.conv2(x)
        back = torch.sub(all, front)
        back = self.div1(back)
        front = self.div2(front)
        return torch.div(front, back)


def cfar(arg):
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

    out = model(
        torch.from_numpy(
            np.expand_dims(
                inputImg.astype(np.float32), 
                axis=[0,1]
            )
        ).to(device)
    )

    img = out.detach().cpu().numpy()[0,0,:,:]
    img = (img > 1) * 255

    # crop
    estimateImg[2*bc:inputImg.shape[0]-2*bc, 2*bc:inputImg.shape[1]-2*bc] = \
        img[2*bc:inputImg.shape[0]-2*bc, 2*bc:inputImg.shape[1]-2*bc]
    # estimateImg = img

    # output
    tmpName = OUTPUT_IMG_DIR + f"{out_name}_{gc}_{bc}_{al}.png"
    cv2.imwrite(tmpName, estimateImg)


if __name__ == '__main__':
    import numpy as np
    import cv2, os
    from tic import Tic
    from utils import getFiles

    modelDir = 'model/sarf.model'
    if not os.path.isdir('model'):
        os.mkdir('model')

    if not os.path.isdir(OUTPUT_IMG_DIR):
        os.mkdir(OUTPUT_IMG_DIR)

    # if os.path.isfile(modelDir):
    #     model = torch.load(modelDir)
    # else:
    #     model = Sarf(GUARD_CELLS, BG_CELLS, ALPHA).to(device)
    #     torch.save(model, modelDir)
    model = Sarf(GUARD_CELLS, BG_CELLS, ALPHA).to(device)

    imgs = getFiles(root, '.jpg')
    for img_path in imgs:
        print(img_path)
        Tic.tic()
        cfar({'img_path':img_path, 'gc':GUARD_CELLS, 'bc':BG_CELLS, 'al':ALPHA})
        Tic.toc()
