# myfirst-neural-network-regressor

This is a simple implementation of two-layered perceptron regressor

## Data
- data are 10000 random numbers from range `[-π, π]`
- data was generated with `generate_data.py` script
- size of the test set: `0.2`
- the target function is `sin(x)`

## Experiments in main.py:
- the script creates 30 regressors and teaches each of them on the same data
- then MAE is calculated and printed in console for each regressor
- the output on test data is visualized on charts in `charts` directory
- the randomness is removed (a seed is set)