# Neural Networks tutorial notes
This repository contains a multi-layer perceptron (3 layer fixed) implementation
with experiments wrapped in a ipython notebook. Some notes and summaries were
included as archive for personal learning purposes, however these notes are not
necessary to reproduce the MLP implementation + experiments.

## MLP implementation
The MLP is a 3 layer MLP that can learn both regression and classification 
problems. It has option to train using stochastic gradient descent and ensemble.
Usage examples can be found inside the notebook.

The source code for this implementation is in `mlp.py`

## Experiments in the notebook
3 experiments are performed in the notebook.

- XOR
- sine function over linear combination of 4 variables
- hand written letter prediction (engineered feature)

Different parameter configuration and their corresponding results are included 
for comparison