# from genericpath import isdir
import torch
from torch import nn
import cv2, os
import numpy as np
# from torch._C import uint8
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda, Compose
# import matplotlib.pyplot as plt

from utils import getFiles, get_yaml_data, dict_merge
arg = get_yaml_data('config/arg.yaml')

device = "cuda" if torch.cuda.is_available() and arg.get('useGPU', False) else "cpu"
print("Using {} device".format(device))


def measureCommand(fun):
    if arg.get('measureCommand', False):
        from tic import Tic
        def measure(*args, **kwargs):
            Tic.tic()
            out = fun(*args, **kwargs)
            Tic.toc()
            return out
        return measure
    else:
        return fun


# Define model
class Carf(nn.Module):
    def __init__(self, arg):
        super(Carf, self).__init__()

        GUARD_CELLS = arg.get('GUARD_CELLS')
        BG_CELLS    = arg.get('BG_CELLS')
        ALPHA       = arg.get('ALPHA')

        CFAR_UNITS  = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)
        HALF_CFAR_UNITS = GUARD_CELLS + BG_CELLS

        self.areaFront = ( (GUARD_CELLS * 2 + 1) **2 ) * ALPHA
        self.areaBack  = CFAR_UNITS ** 2 - (GUARD_CELLS * 2 + 1) ** 2

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


    def forward(self, x):
        all   = self.conv1(x)
        front = self.conv2(x)
        back  = torch.sub(all, front)
        front = torch.div(front, self.areaFront)
        back  = torch.div(back, self.areaBack)
        return torch.cat([torch.div(front, back), front], dim=0)

@measureCommand
def cfar(arg):
    img_path    = arg.get('img_path')
    gc          = arg.get('GUARD_CELLS')
    bc          = arg.get('BG_CELLS')
    al          = arg.get('ALPHA')
    inputImg    = arg.get('inputImg')
    inShape     = inputImg.shape[:2]
    out_name    = os.path.basename(img_path).split('.')[0]
    estimateImg = np.zeros((inShape[0], inShape[1]), np.uint8)
    CFAR_UNITS  = 1 + (gc * 2) + (bc * 2)
    HALF_CFAR_UNITS = gc + bc
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

    out = out.detach().cpu().numpy()
    img = out[0,0,:,:]
    front = out[1,0,:,:]
    img = (img > 1) * (front > 25) * 255

    # crop
    if arg.get('cropEdge'):
        estimateImg[2*bc:inShape[0]-2*bc, 2*bc:inShape[1]-2*bc] = \
            img[2*bc:inShape[0]-2*bc, 2*bc:inShape[1]-2*bc]
    else:
        estimateImg = img

    # output
    if arg.get('saveResult', False) is True and arg.get('OUTPUT_IMG_DIR', None) is not None:
        tmpName = os.path.join(arg.get('OUTPUT_IMG_DIR'), f"{out_name}_{gc}_{bc}_{al}_{inShape}.png")
        cv2.imwrite(tmpName, estimateImg)

    return estimateImg


if __name__ == '__main__':

    from image_pyramid import ImagePyramid

    OUTPUT_IMG_DIR = arg.get('OUTPUT_IMG_DIR')
    if not os.path.isdir(OUTPUT_IMG_DIR):
        os.mkdir(OUTPUT_IMG_DIR)

    model = Carf(arg).to(device).eval()

    img_paths = getFiles(arg.get('root'), '.jpg')
    for img_path in img_paths:
        print('\t', img_path)
        inputImg = cv2.imread(img_path, 0).astype(float)
        for i, inputImg in enumerate(ImagePyramid()(inputImg, arg.get('ratio',0.7071), arg.get('layers',2))):
            cfar(dict_merge(arg, {'img_path':img_path, 'inputImg':inputImg, 'model':model}))
