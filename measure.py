def measureCommand(fun):
    
    from tic import Tic

    def measure(*args, **kwargs):
        if args[0].get('measureCommand', False): Tic.tic()
        out = fun(*args, **kwargs)
        if args[0].get('measureCommand', False): Tic.toc()
        return out

    return measure