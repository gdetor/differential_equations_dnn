# This script contains auxiliary functions used by neural_field_temporal
# and neural_field_spatiotemporal scripts.
# Copyright (C) 2023  Georgios Is. Detorakis (gdetor@protonmail.com)
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.
import sys
import time
from functools import wraps


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print(f"Total time running {function.__name__}: {str(t1-t0)}")
        return result
    return function_timer


def parameters_summary(fname,
                       activation,
                       source,
                       k,
                       nn_type,
                       hidden_size,
                       num_layers,
                       batch_size,
                       num_train_samples,
                       num_test_samples,
                       epochs,
                       xmin,
                       xmax,
                       tf,
                       case=["spatiotemporal", "1d"]):
    """!
    @param fname (str) Core filename where to print the summary
    @param activation_func (str)    Type of neural field's activation function
    @param source (str)  Type of input source for the neural field
    @param k (int)  Number of spatial nodes of the neural field
    @param nn_type (str) Type of neural network (MLP or DGM)
    @param hidden_size (int)  Neural network's number of hidden units
    @param num_layers (int)  Neural network's number of hidden layers
    @param batch_size (int) Batch size
    @param num_train_samples (int) Number of training samples
    @param num_test_samples (int) Number of test samples
    @param epochs (int) Number of epochs
    @param xmin (float) Lower bound of Omega
    @param xmax (float) Upper bound of Omega
    @param tf (float) The total number of time points for which to solve for u
    @param case Determines one- or two-dimensional neural field cases for
    storing purposes

    @return Void
    """
    original_stdout = sys.stdout
    fname_ = fname+"_"+case[0]+"_"+case[1]+".pms"
    with open(fname_, "w") as f:
        sys.stdout = f
        print("------------------------------------------------------")
        print(f"Neural network type: {nn_type}")
        print(f"Activation function: {activation}")
        print(f"Source type: {source}")
        print(f"Neural field spatial discrete nodes (neurons): {k}")
        print(f"Number of hidden layers : {num_layers}")
        print(f"Number of hidden layer neurons: {hidden_size}")
        print(f"Batch size: {batch_size}")
        print(f"#Train samples: {num_train_samples}")
        print(f"#Test samples: {num_test_samples}")
        print(f"Epochs: {epochs}")
        print(f"xmin: {xmin}")
        print(f"xmax: {xmax}")
        print(f"tf: {tf}")
        print("------------------------------------------------------")
        sys.stdout = original_stdout
