# Solving Differential Equations with Deep Neural Networks


This repository provides some basix examples on how to use deep neural networks,
and more precisely feed-forward and LSTM-like networks, to solve ordinary 
differential equations (ODEs), partial differential equations (PDEs), and
integral equations. The neural networks are used to approximate the solution
of a differential equation based on the Deep Galerkin method proposed by [1].
More precisely,
  - **simple_ode** solves a first order linear ordinary differential equation.
  - **heat.py** solves a one-dimensional heat equation (PDE).
  - **fredholm.py** solves a second kind Fredholm integral equation.
  - **fitzhugh_nagumo.py** solves the Fitzhugh-Nagumo system of ODEs.
  - **neural_field.py** This script shows how to solve a one-dimensional neural
  field.
All the Python scripts in this repository consist supplementary material to 
[2].


## How to get the scripts

To get the scripts you can just clone the repository to your computer by executing the 
following commands:

```bash
git clone differential_equations_dnn
cd differential_equations_dnn
```

## How to run the scripts

To run the scripts you can just type the following command:
```bash
$ python3 heat.py --solve --plot --savefig
```
The arguments `--solve` and `--plot` call the methods that solve the heat
equation and plot the results, respectively. The `--savefig` argument stores
the plotted figure to the directory `figs/`.


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
  3.
