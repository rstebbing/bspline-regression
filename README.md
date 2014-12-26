bspline-regression
==================

<p align="center">
  <img src="https://github.com/rstebbing/bspline-regression/raw/master/figures/README-0.png" alt="B-spline regression"/>
</p>

This repository provides the code necessary to fit uniform B-splines of arbitrary degree to 2D/3D point data.
For example, the noisy red points above are approximated by a uniform cubic B-spline (blue) with 14 control points (black).

The primary purpose for making this repository public is to provide a general B-spline solver to developers which is easy to use, has minimal dependencies, but is still performant.
The secondary purpose is to provide a starting point for people learning about B-splines, sparse non-linear least squares optimisation, and the Levenberg-Marquardt algorithm in particular.

Author: Richard Stebbing

License: MIT (refer to LICENSE)

Dependencies
------------
* Numpy
* Scipy
* Sympy
* matplotlib

Examples
--------

### 2D
To generate the above example problem and initialisation:
```
python generate_example.py 512 1 1e-1 3 14 Example_1.json --seed=0 --frequency=3.0
```
The arguments passed to [generate_example.py](generate_example.py) here are:

Value | Argument | Description
------------- | ------------- | -------------
`512` | `num_data_points` | The number of data points to generate.
`1` | `w` | The weight applied to each squared residual.
`1e-1` | `lambda_` | The regularisation weight.
`3` | `degree` | The degree of the uniform B-spline.
`14` | `num_control_points` | The number of control points.
`Example_1.json` | `output_path` | The output path.
`--seed` | `0` | The random number generator seed (optional).
`--frequency` | `3.0` | The frequency of the `sin` (optional).

Additionally:
* `w` can be a pair (2D) or triple (3D) so that `w[i]` is the weight applied for the squared residuals in the `i`th dimensions.
This is useful for time-series data where the uncertainty of the measurement (y-axis) is much greater than that of the reported time (x-axis).
* Increasing `lambda_` increases the "force" pulling adjacent B-spline control points together.

The output `Example_1.json` is a dictionary of the form:

Key | Value Type | Description
------------- | ------------- | -------------
`degree` | `int` | The degree of the uniform B-spline.
`num_control_points` | `num_control_points` | The number of control points.
`dim` | `int` | The dimension of the problem.
`is_closed` | `bool` | `True` if the B-spline is closed, `False` otherwise.
`Y` | `array_like, shape = (num_data_points, dim)` | The array of data points.
`w` | `array_like, shape = (num_data_points, dim)` | The weight of each residual on each dimension.
`lambda_` | `float` | The regularisation weight.
`X` | `array_like, shape = (num_control_points, dim)` | The array of B-spline control points.
`u` | `array_like, shape = (num_data_points, 1)` | The array of correspondences (*preimages*).

In summary, the first four keys define the *structure* of the B-spline, the next three specify the data points and problem configuration, and the final two define the *state*: `X` is the array of control points which defines the geometry of the B-spline and `u` is the array of *preimages*.
That is, `u[i]` gives the *coordinate* of the point on the B-spline that is closest to `Y[i]`.
In [generate_example.py](generate_example.py), `X` is initialised to a straight line and `u` is set approximately.

Visualising the initialisation is straightforward:
```
python visualise.py Example_1.json
```
<p align="center">
  <img src="https://github.com/rstebbing/bspline-regression/raw/master/figures/README-1.png" alt="Initialisation"/>
</p>
where correspondences are shown in orange.

To solve for `X` and `u` and visualise the output:
```
python uniform_bspline_regression.py Example_1.json Example_1_Output.json
python visualise.py Example_1_Output.json
```

Alternatively, to save and visualise *all* optimisation steps:
```
python uniform_bspline_regression.py Example_1.json Example_1_Output --output-all
python visualise.py Example_1_Output Example_1_Output_Visualisation --empty
```
Additional arguments to [visualise.py](visualise.py) and [uniform_bspline_regression.py](uniform_bspline_regression.py) can be found with `--help`.

```
python generate_example.py 512 "1e1,1" 1e-1 5 16 512_1e1_1_1e-1_5_16_2.json --seed=0 --frequency=0.5
python uniform_bspline_regression.py 512_1e1_1_1e-1_5_16_2.json 512_1e1_1_1e-1_5_16_2-1.json
python visualise.py 512_1e1_1_1e-1_5_16_2-1.json
```
