<img src="./docs/poppy_logo.png" width="100">

# PoPPy
A Point Process Toolbox Based on PyTorch

**PoPPy** is a machine learning toolbox focusing on point process model, which achieves rich functionality on data operations, high flexibility on model design and high scalability on optimization.


## Data operations

* Import event sequences and their features from csv files
* Random and/or feature-based event sequence stitching
* Random and/or feature-based event sequence superposing
* Event sequence aggregating
* Batch sampling of event sequence


## Models

* Poisson process
* linear Hawkes process
* nonlinear Hawkes process
* Factorized point process
* Feature-involved point process
* Mixture models of point processes


## Loss functions

* Maximum likelihood estimation
* Least-square estimation
* Conditional likelihood estimation


## Decay kernels

For Hawkes processes, multiple decay kernels are applicable:
* Exponential kernel
* Rayleigh kernel
* Gaussian kernel
* Gate kernel
* Powerlaw kernel
* Multi-Gaussian kernel


## Optimization

* SGD based on learning algorithms
* Support CPU or GPU-based computations


## Simulation
* Ogata's thinning algorithm


## Installation
* Step 1: Install Anaconda3 and PyTorch 0.4
* Step 2: Download the package and unzip it
* Step 3: Change "POPPY_PATH" in dev/util.py to the directory of the downloaded package.

## Usage
More details can be found in [tutorial](https://arxiv.org/pdf/1810.10122.pdf) and the pdf files in the folder "docs". 

## On going 
* Debugging GPU version demo
* Implementing mixture models of point processes
* Adding more examples

