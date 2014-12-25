# visualise.py

# Imports
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from uniform_bspline import Contour


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('--num-samples', type=int, default=1024)
    args = parser.parse_args()

    print 'Input:', args.input_path
    with open(args.input_path, 'rb') as fp:
        z = json.load(fp)

    degree, num_control_points, dim, is_closed = (
        z['degree'], z['num_control_points'], z['dim'], z['is_closed'])

    print '  degree:', degree
    print '  num_control_points:', num_control_points
    print '  dim:', dim
    print '  is_closed:', is_closed
    c = Contour(degree, num_control_points, dim, is_closed=is_closed)

    Y, w, u, X = map(np.array, [z['Y'], z['w'], z['u'], z['X']])
    print '  num_data_points:', Y.shape[0]

    kw = {}
    if Y.shape[1] == 3:
        kw['projection'] = '3d'
    f = plt.figure()
    ax = f.add_subplot(111, **kw)
    ax.set_aspect('equal')
    def plot(X, *args, **kwargs):
        ax.plot(*(tuple(X.T) + args), **kwargs)

    plot(Y, 'ro')

    for m, y in zip(c.M(u, X), Y):
        plot(np.r_['0,2', m, y], 'k-')

    plot(X, 'bo--', ms=8.0)
    plot(c.M(c.uniform_parameterisation(args.num_samples), X), 'b-', lw=2.0)

    plt.show()


if __name__ == '__main__':
    main()
