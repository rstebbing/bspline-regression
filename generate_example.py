##########################################
# File: generate_example.py              #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import argparse
import json
import numpy as np
import scipy.spatial

from uniform_bspline import Contour

# float_tuple
def float_tuple(s):
    return tuple(float(f) for f in s.split(','))


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_data_points', type=int)
    parser.add_argument('w', type=float_tuple)
    parser.add_argument('lambda_', type=float)
    parser.add_argument('degree', type=int)
    parser.add_argument('num_control_points', type=int)
    parser.add_argument('output_path')
    parser.add_argument('--alpha', type=float, default=1.0 / (2.0 * np.pi))
    parser.add_argument('--dim', type=int, choices={2, 3}, default=2)
    parser.add_argument('--frequency', type=float, default=1.0)
    parser.add_argument('--num-init-points', type=int, default=16)
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    print 'Parameters:'
    print '  alpha:', args.alpha
    print '  frequency:', args.frequency
    x = np.linspace(0.0, 2.0 * np.pi, args.num_data_points)
    y = np.exp(-args.alpha * x) * np.sin(args.frequency * x)

    if args.dim == 2:
        Y = np.c_[x, y]
    else:
        Y = np.c_[x, y, np.linspace(0.0, 1.0, args.num_data_points)]

    x0, x1 = Y[0].copy(), Y[-1].copy()
    t = np.linspace(0.0, 1.0, args.num_control_points)[:, np.newaxis]
    X = x0 * (1 - t) + x1 * t

    c = Contour(args.degree, args.num_control_points, args.dim)
    m0, m1 = c.M(c.uniform_parameterisation(2), X)
    x01 = 0.5 * (x0 + x1)
    X = (np.linalg.norm(x1 - x0) / np.linalg.norm(m1 - m0)) * (X - x01) + x01

    if args.seed is not None:
        np.random.seed(args.seed)
    print '  sigma:', args.sigma
    Y += args.sigma * np.random.randn(Y.size).reshape(Y.shape)

    if np.any(np.asarray(args.w) < 0):
        raise ValueError('w <= 0.0 (= {})'.format(args.w))
    if len(args.w) == 1:
        w = np.empty((args.num_data_points, args.dim), dtype=float)
        w.fill(args.w[0])
    elif len(args.w) == args.dim:
        w = np.tile(args.w, (args.num_data_points, 1))
    else:
        raise ValueError('len(w) is invalid (= {})'.format(len(args.w)))

    if args.lambda_ <= 0.0:
        raise ValueError('lambda_ <= 0.0 (= {})'.format(args.lambda_))

    u0 = c.uniform_parameterisation(args.num_init_points)
    D = scipy.spatial.distance.cdist(Y, c.M(u0, X))
    u = u0[D.argmin(axis=1)]

    to_list = lambda _: _.tolist()
    z = dict(degree=args.degree,
             num_control_points=args.num_control_points,
             dim=args.dim,
             is_closed=False,
             Y=to_list(Y),
             w=to_list(w),
             lambda_=args.lambda_,
             u=to_list(u),
             X=to_list(X))

    print 'Output:', args.output_path
    with open(args.output_path, 'wb') as fp:
        fp.write(json.dumps(z, indent=4))


if __name__ == '__main__':
    main()
