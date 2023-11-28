# Introduction

This is my code implementation of "MatDot" proposed in paper *[On the Optimal Recovery Threshold of Coded Matrix Multiplication](https://arxiv.org/abs/1801.10292)*.

This repo is a new implementaion, with cleaner code, modular functions and user-friendly nameing.

Note: I still have several problems understanding the paper and the [original implementation](https://github.com/nitishmital/distributed_matrix_computation/blob/master/MatDot.sage). I don't know the meaning of the interpolation part and why aren't the sizes of input $A,B$  square.

# How to Start

This code uses `sage` python library.
Here's instructions to run it:

```shell
sudo conda create -n sage sage python=3.9
conda activate sage
python ./MatDot.py
```

