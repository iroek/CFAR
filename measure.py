from utils import get_yaml_data
measure = get_yaml_data('config/arg.yaml').get('measureCommand', True)

def measureCommand(fun):
    
    from tic import Tic

    def measure(*args, **kwargs):
        if measure: Tic.tic()
        out = fun(*args, **kwargs)
        if measure: Tic.toc()
        return out

    return measure