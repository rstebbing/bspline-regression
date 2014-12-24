# util.py

# raise_if_not_shape
def raise_if_not_shape(name, A, shape):
    if A.shape != shape:
        raise ValueError('{}.shape != {}'.format(name, shape))
