# util.py

# raise_if_not_shape
def raise_if_not_shape(name, A, shape):
    """Raise a `ValueError` if the np.ndarray `A` does not have dimensions
    `shape`."""
    if A.shape != shape:
        raise ValueError('{}.shape != {}'.format(name, shape))
