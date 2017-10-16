import pytest
import mxnet as mx
import numpy as np
import yaml

from mxnet import gluon
from mxnet import nd
from mxnet import autograd

import deep_exp_fam
import distributions
import tester


def test_latent_gaussian_layer():
  """Test matching q(z) to p(z) where p is Gaussian."""
  mean = 3.9

  config = """
  n_iterations: 100
  learning_rate: 0.1
  gradient:
    estimator: score_function
    n_samples: 16
    batch_size: 1
  layer_1:
    latent_distribution: gaussian
    p_z_mean: {}
    p_z_variance: 1.
    size: 1
  """.format(mean)

  def test_posterior_predictive(sample: np.array) -> None:
    print('posterior predictive sample:', sample)
    print('prior mean:', mean)
    np.testing.assert_allclose(sample, mean, rtol=1e-1)

  tester.test(config, data=np.array([np.nan]),
              test_fn=test_posterior_predictive)


def test_latent_poisson_layer():
  """Test matching q(z) to p(z) where p is Poisson."""
  mean = 5.

  config = """
  n_iterations: 100
  learning_rate: 0.1
  gradient:
    estimator: score_function
    n_samples: 16
    batch_size: 1
  layer_1:
    latent_distribution: poisson
    size: 1
    p_z_mean: {}
  """.format(mean)

  def test_posterior_predictive(sample: np.array) -> None:
    print('posterior predictive sample:', sample)
    print('prior mean:', mean)
    np.testing.assert_allclose(sample, mean, rtol=1e-1)

  tester.test(config, data=np.array([np.nan]),
              test_fn=test_posterior_predictive)
