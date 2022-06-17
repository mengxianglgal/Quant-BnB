# Quant-BnB
### Rahul Mazumder, Xiang Meng, Haoyue Wang
### Massachusetts Institute of Technology

## Introduction
Quant-BnB is a discrete optimization method for solving optimal decision trees (i.e.,with smallest training error). More concretely, Quant-BnB is a novel branch-and-bound (BnB) algorithm for the computation of shallow optimal trees (i.e. depth=$2,3$). It can handle both classification and regression tasks and is designed to directly handle continuous features (including a mix of continuous and binary features). 

To our knowledge, Quant-BnB is the first standalone method (i.e, does not rely on proprietary optimization solvers) for optimal classification/regression trees that directly applies to datasets with continuous features. It achieves significant empirical improvements (10x~100x) compared to existing methods. and can scale to problems where the number of data points n ~ 10^5.

## Installation
The toolkit is implemented in Julia. To run Quant-BnB, simply include the code by:
```julia
include("QuantBnB-2D.jl")
include("QuantBnB-3D.jl")
```

## Structure of the repo

* `QuantBnB-2D.jl` implements Quant-BnB for trees with depth 2.
* `QuantBnB-3D.jl` implements Quant-BnB for trees with depth 3.
* `Algorithms.jl`, `gen_data.jl` and `lowerbound_middle.jl`  contains some auxiliary functions.
* `test.ipynb` contains examples of Quant-BnB on various tasks.
* `dataset/` contains classification and regression datasets.
