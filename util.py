##########################################
# File: util.py                          #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import re


# raise_if_not_shape
def raise_if_not_shape(name, A, shape):
    """Raise a `ValueError` if the np.ndarray `A` does not have dimensions
    `shape`."""
    if A.shape != shape:
        raise ValueError('{}.shape != {}'.format(name, shape))


# previous_float
PARSE_FLOAT_RE = re.compile(r'([+-]*)0x1\.([\da-f]{13})p(.*)')
def previous_float(x):
    """Return the next closest float (towards zero)."""
    s, f, e = PARSE_FLOAT_RE.match(float(x).hex().lower()).groups()
    f, e = int(f, 16), int(e)
    if f > 0:
        f -= 1
    else:
        f = int('f' * 13, 16)
        e -= 1
    return float.fromhex('{}0x1.{:013x}p{:d}'.format(s, f, e))
