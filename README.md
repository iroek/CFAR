# SUMMARY

If have any CUDA device, use cfar_acc_torch.py for more speed at about 1.5s/it on 2080ti when processing 2K picture, 

otherwise use cfar_acc_single-file.py, performs less than 20s/it on i9-9900k.

refer to demo.ps1 and cfar_acc_torch.py for CFAR demo

refer to region_growing.py for detecting demo

# FILES

| File                    | Describtion                          |
| ----------------------- | ------------------------------------ |
| cfar.py                 | copied from CSDN                     |
| cfar_acc_multi-file.py  | Cfar more than one image in parallel |
| cfar_acc_single-file.py | cfar one image in parallel           |
| cfar_acc_torch.py       | cfar accelerate by torch on GPU      |
| cfar_once.py            | cfar at one position                 |
| image_pyramid.py        | tool for creating Image Pyramid      |
| measure.py              | decorator for task time measuring    |
| region_growing.py       | detect method of Region Growing      |
| tic.py utils.py         | other tools                          |

