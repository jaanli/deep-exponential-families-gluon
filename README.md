### Deep exponential families (gluon / mxnet)

This is an implementation of deep exponential families in gluon (mxnet). DEFs are described in https://arxiv.org/abs/1411.2581

I found it much easier to implement this in an imperative / dynamic graph library like mxnet than in autodifferentiation libraries that only support static computation graphs.

Currently the code only implements a point-mass distributions for the weights and biases of each layer in the DEF (these parameters are learned using variational expectation-maximization). It should be straightforward to extend this to other distributions.

The gradients are computed with either the score function estimator or the pathwise (reparameterization trick) estimator. For score function gradient estimators, we use the optimal control variate scaling described in [black box variational inference](https://arxiv.org/abs/1401.0118).

The code takes lots of inspiration from the official deep exponential families [codebase](https://github.com/blei-lab/deep-exponential-families) and the gluon examples in mxnet.

### Example

Train a Poisson deep exponential family model on a large collection of science articles (in the LDA-C format):
```
PYTHONPATH=. python experiments/poisson_gaussian_deep_exp_fam_text.py
```
This periodically prints out the latent factors (dimensions of the latent variable), and the weight associated with each. For example, a dimension captures documents about DNA:
```
0.246	fig
-0.358	dna
-0.366	protein
-0.372	cells
-0.430	cell
-0.722	gene
-0.970	binding
-1.010	two
-1.026	sequence
-1.100	proteins
```

To train a Poisson deep exponential family model on the MNIST dataset:
```
PYTHONPATH=. python experiments/poisson_gaussian_deep_exp_fam_mnist.py
```

Also see examples in `tests/` folder.

### Requirements
Install requirements with [anaconda](https://conda.io/docs/user-guide/install/index.html):
```
conda env create -f environment.yml
source activate deep_exp_fam
```

### Testing
Run `PYTHONPATH=. pytest` for unit tests and `mypy $(find . -name '*.py')` for static type-checking.

### TODO:
* figure out a cleaner way to do per-sample gradients -- bug tracker: https://github.com/apache/incubator-mxnet/issues/7987 (right now, parameters are repeated in deep_exp_fam.DeepExponentialFamilyModel class and require annoying processing)
* add support for priors on the weights
