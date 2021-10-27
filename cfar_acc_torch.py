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


# Define model
class Sarf(nn.Module):
    def __init__(self, arg):
        super(Sarf, self).__init__()

        GUARD_CELLS = arg.get('GUARD_CELLS')
        BG_CELLS    = arg.get('BG_CELLS')
        ALPHA       = arg.get('ALPHA')

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
    gc          = arg.get('GUARD_CELLS')
    bc          = arg.get('BG_CELLS')
    al          = arg.get('ALPHA')
    inputImg    = cv2.imread(img_path, 0).astype(float)
    out_name    = os.path.basename(img_path).split('.')[0]
    estimateImg = np.zeros((inputImg.shape[0], inputImg.shape[1]), np.uint8)
    CFAR_UNITS  = 1 + (gc * 2) + (bc * 2)
    HALF_CFAR_UNITS = gc + bc
    arg['inputImg'] = inputImg
    arg['CFAR_UNITS']      = CFAR_UNITS
    arg['HALF_CFAR_UNITS'] = HALF_CFAR_UNITS

    model = arg.get('model')
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
    if arg.get('cropEdge'):
        estimateImg[2*bc:inputImg.shape[0]-2*bc, 2*bc:inputImg.shape[1]-2*bc] = \
            img[2*bc:inputImg.shape[0]-2*bc, 2*bc:inputImg.shape[1]-2*bc]
    else:
        estimateImg = img

    # output
    if arg.get('saveResult'):
        tmpName = os.path.join(arg.get('OUTPUT_IMG_DIR'), f"{out_name}_{gc}_{bc}_{al}.png")
        cv2.imwrite(tmpName, estimateImg)

    return estimateImg


if __name__ == '__main__':
    import numpy as np
    import cv2, os
    from tic import Tic
    from utils import getFiles, get_yaml_data

    arg = get_yaml_data('config/arg.yaml')

    OUTPUT_IMG_DIR = arg.get('OUTPUT_IMG_DIR')
    if not os.path.isdir(OUTPUT_IMG_DIR):
        os.mkdir(OUTPUT_IMG_DIR)

    model = Sarf(arg).to(device).eval()

    img_paths = getFiles(arg.get('root'), '.jpg')
    for img_path in img_paths:
        print(img_path, end=' ')
        Tic.tic()
        arg.update({'img_path':img_path, 'model':model})
        cfar(arg)
        Tic.toc()
