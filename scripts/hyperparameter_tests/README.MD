The scripts in this file were used to test the efect of different hyper parameters on the results of training

All files were generated using src/generate_random_params.py

Hyperparameters tested:
- learning rate = 1e-4
    - sample from [1e-5, 1e-2]
- regularization (dropout=0.2)
    - sample from [0.1, 0.4]
- weight decay (currently pytorch default = 1e-2)
    - sample from [1e-4, 1e-1]