# Solving Differential Equations with Deep Neural Networks


This repository provides some basic examples of using deep neural networks and feed-forward and LSTM-like neural networks to solve ordinary differential equations (ODEs), partial differential equations (PDEs), and integral equations. Neural networks, in general, can approximate the solution of a differential equation based on the Deep Galerkin method proposed by [1].

The present repository contains the following Python scripts:
  - **simple_ode** solves a first-order linear ordinary differential equation.
  - **heat.py** solves a one-dimensional heat equation (PDE).
  - **fredholm.py** solves a Fredholm integral equation of the second kind.
  - **fitzhugh_nagumo.py** solves the Fitzhugh-Nagumo system of ODEs.
  - **neural_field.py** This script shows how to solve a one-dimensional neural field.
  - **optimize_heat_ray.py** This script uses Ray Tune to search for optimal hyperparameters.
  - **batchsize_effect_heat.py** This script shows how different batch sizes can affect the minimization process when we approximate the solution of a heat equation.
  - **batchnorm_effect_heat.py** Similar to the previous script, this one shows how batch normalization affects the minimization process of approximating the solution of a one-dimensional heat equation.
All the Python scripts in this repository are supplementary material [2]. They are meant for education purposes only, and thus, the code is not organized for production.


## How to get the scripts

To get the scripts, you can just clone the repository to your computer by executing the following commands:

```bash
git clone differential_equations_dnn
cd differential_equations_dnn
```

## How to run the scripts

To run the scripts you can just type the following command:
```bash
$ python3 heat.py --solve --plot --savefig --niters 15000 --nnodes 40
```
The arguments `--solve` and `--plot` call the methods that solve the heat equation and plot the results, respectively.
The arguments `--niters` and `nnodes` are the number of iterations used to minimize the loss function and the number of discretization nodes used to evaluate the solution's approximation, respectively.
The `--savefig` argument stores the plotted figure in the directory `figs/`.


If the user wants to run the Ray hyperparameters tune script, they should run the following:
```bash
$ python3 optimize_heat_ray.py
```
After a few minutes, the user will get the optimal hyperparameters that minimize the loss function once the optimization process is complete. An example of the output is given below:
```bash
{'batch_size': 214, 'n_iters': 30348, 'lrate': 0.00020380637765567638}
```

The rest of the scripts expect no arguments. Therefore, the user can run them as usual.


## Dependencies

The Python scripts require the following Python packages to be able to run:

  - Numpy 1.26.4
  - Matplotlib 3.5.1
  - Torch 2.0.0 

## Tested platforms 

The software available in this repository has been tested on the following platforms:
  - Ubuntu 22.04.4 LTS
  - Python 3.10.12
  - GCC 11.4.0
  - x86_64

## References
  1. J. Sirignano and K. Spiliopoulos, "DGM: A deep learning algorithm for
    solving partial differential equations", 2018.
  2. G. Is. Detorakis, *Practical Aspects on Solving Differential Equations Using Deep
  Learning: A Primer*, 2024.
