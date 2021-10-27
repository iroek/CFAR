import numpy as np

def cfar_one(arg):
    i,j             = arg.get('index')
    gc              = arg.get('gc')
    bc              = arg.get('bc')
    al              = arg.get('al')
    inputImg        = arg['inputImg']
    CFAR_UNITS      = arg['CFAR_UNITS']
    HALF_CFAR_UNITS = arg['HALF_CFAR_UNITS']

    average = (inputImg[i:i+CFAR_UNITS, j:j+CFAR_UNITS].sum() - inputImg[i+bc:i+bc+2*gc+1, j+bc:j+bc+2*gc+1].sum()) \
            / (CFAR_UNITS ** 2 - ((gc * 2) + 1) **2 )

    center_cell_x = i + HALF_CFAR_UNITS
    center_cell_y = j + HALF_CFAR_UNITS
    if inputImg[center_cell_x, center_cell_y] > (average * al):
        return (center_cell_x, center_cell_y, 255)
    else:
        return (center_cell_x, center_cell_y, 0)

def cfar_one_improved(arg):
    
    i,j             = arg.get('index')
    gc              = arg.get('GUARD_CELLS')
    bc              = arg.get('BG_CELLS')
    al              = arg.get('ALPHA')
    inputImg        = arg['inputImg']
    CFAR_UNITS      = arg['CFAR_UNITS']
    HALF_CFAR_UNITS = arg['HALF_CFAR_UNITS']

    average = (inputImg[i:i+CFAR_UNITS, j:j+CFAR_UNITS].sum() - inputImg[i+bc:i+bc+2*gc+1, j+bc:j+bc+2*gc+1].sum()) \
            / (CFAR_UNITS ** 2 - ((gc * 2) + 1) **2 )

    front   = inputImg[i+bc:i+bc+2*gc+1, j+bc:j+bc+2*gc+1].sum() / ((gc * 2) + 1) **2 

    center_cell_x = i + HALF_CFAR_UNITS
    center_cell_y = j + HALF_CFAR_UNITS
    if front > (average * al):
        return (center_cell_x, center_cell_y, 255)
    else:
        return (center_cell_x, center_cell_y, 0)

def dict_merge(dict1, dict2):
    dict2.update(dict1)
    return(dict2) 