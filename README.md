python generate_example.py 5 16 512 5_16_512_2.json --seed=0 --frequency=2
python uniform_bspline_regression.py 5_16_512_2.json 1e-1 5_16_512_2_1e-1.json
python visualise.py 5_16_512_2_1e-1.json

python generate_example.py 3 14 512 3_14_512_3_3.json --seed=0 --frequency=3 --dim=3
python uniform_bspline_regression.py 3_14_512_3_3.json 1e-1 3_14_512_3_3_1e-1.json
python visualise.py 3_14_512_3_3_1e-1.json
