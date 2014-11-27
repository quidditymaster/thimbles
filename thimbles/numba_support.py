from thimbles import _with_numba

class DoubleDoppleganger(object):
    
    def __getitem__(self, index):
        return self
    
    def __call__(self, *args, **kwargs):
        return self

if _with_numba:
    import numba
    from numba import double
else:
    double = DoubleDoppleganger()

def _passthrough(func):
    return func

def jit(*args, **kwargs):
    if _with_numba:
        return numba.jit(*args, **kwargs)
    else:
        return _passthrough
