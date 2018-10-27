<img src="./docs/poppy_logo.png" width="100">

# PoPPy
A Point Process Toolbox Based on PyTorch

**PoPPy** is a machine learning toolbox focusing on point process model, which achieves rich functionality on data operations, high flexibility on model design and high scalability on optimization.


## Recent updates
* PoPPy starts to support GPU-based computation. For feature-based model and large-scale dataset with lots of event types, the learning process is accelerated.  
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


## Platform
* I developed and tested PoPPy on MacOS>=10.13, Ubuntu=16.04LTS, and Windows10 (Conda environment)


## Installation
* Step 1: Install Anaconda3 and PyTorch 1.0
* Step 2: Download the package and unzip it
* Step 3: Change "POPPY_PATH" in dev/util.py to the directory of the downloaded package.


## Usage
More details can be found in [tutorial](https://arxiv.org/abs/1810.10122) and the pdf files in the folder "docs". 

## Citation
@article{xu2018poppy,
  title={PoPPy: A Point Process Toolbox Based on PyTorch},
  author={Xu, Hongteng},
  journal={arXiv preprint arXiv:1810.10122},
  year={2018}
}

## Tricks
* Generally, the parameters of exogenous intensity and those of endogenous impact are not in a same scale, i.e., mu ~ O(1/C) and alpha ~ O(1/C^2). When learning the model, different learning rates should be applied to different parameters adaptively. Therefore, although all SGD optimizers in PyTorch are usable, Adam is the recommended choice.
* When softplus activation is applied to a model, we'd better turn the sparsity and nonnegative contraints off.
* When training the mixture model of Hawkes processes, we need to select a large epoch to get meaningful clustering results. I found that in the initial phase, the distribution of clusters will be very imbalanced, i.e., most of sequences will be categorized into one cluster. Fortunately, with the increase of epochs, the distribution will be rebalanced and the sizes of different clusters will be comparable. 
  

## On going 
* Integrate more advanced models
* Adding more examples
* Documentation
* Optimizing code framework and data structure to achieve further acceleration
