python generate_example.py 512 "1e1,1" 1e-1 5 16 512_1e1_1_1e-1_5_16_2.json --seed=0 --frequency=2.0
python uniform_bspline_regression.py 512_1e1_1_1e-1_5_16_2.json 512_1e1_1_1e-1_5_16_2-1.json
python visualise.py 512_1e1_1_1e-1_5_16_2-1.json

python generate_example.py 512 1 1e-1 3 14 512_1_1e-1_3_14_3.json --seed=0 --frequency=3.0
python uniform_bspline_regression.py 512_1_1e-1_3_14_3.json 512_1_1e-1_3_14_3-1.json
python visualise.py 512_1_1e-1_3_14_3-1.json
