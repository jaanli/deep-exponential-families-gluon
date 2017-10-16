import pytest
import mxnet as mx
import numpy as np
import yaml

from mxnet import gluon
from mxnet import nd
from mxnet import autograd

from deep_exp_fam import deep_exponential_family_model as lib

np.random.seed(32423)
mx.random.seed(3242)


def test_linear_time_estimator():
  """Test the control variate estimator with Rajesh's numpy implementation."""
  init_mean = 3.9
  n_samples = 256
  mean = nd.ones(n_samples) * init_mean
  mean.attach_grad()
  variance = nd.array([1.])
  sample = nd.stop_gradient(
      mean + variance * nd.random_normal(shape=(n_samples,)))
  with autograd.record():
    log_prob = (-0.5 * nd.log(2. * np.pi * variance)
                - 0.5 * nd.square(sample - mean) / variance)
    log_prob.backward()
  f = nd.square(sample) - 3.
  h = mean.grad
  grad = lib._leave_one_out_gradient_estimator(h, f)
  fast_grad = lib._leave_one_out_gradient_estimator(h, f, zero_mean_h=True)
  np_grad = np.mean(leave_one_out_control_variates(h.asnumpy(), f.asnumpy()))
  np.testing.assert_allclose(grad.asnumpy(), np_grad, rtol=1e-3)
  np.testing.assert_allclose(fast_grad.asnumpy(), np_grad, rtol=1e-2)


def test_held_out_covariance():
  """Test leave-one-out covariance estimation."""
  x = np.random.rand(10)
  y = np.random.rand(10)
  cov = lib._held_out_covariance(nd.array(x), nd.array(y))
  np_cov = held_out_cov(x, y)
  np.testing.assert_allclose(cov.asnumpy(), np_cov, rtol=1e-2)


def leave_one_out_control_variates(score, f):
  held_out_covariance = held_out_cov(score, f * score)
  held_out_variance = held_out_cov(score, score)
  optimal_a = held_out_covariance / held_out_variance
  grad = score * (f - optimal_a)
  return grad


def held_out_cov(x, y):
  n = len(x)
  C = np.cov(x, y)[0, 1]  # Slightly wasteful
  meanx = np.mean(x)
  meany = np.mean(y)
  C *= (n - 1)

  meanx_ho = (meanx - x / n) / (1 - 1.0 / n)
  meany_ho = (meany - y / n) / (1 - 1.0 / n)
  C -= (x - meanx_ho) * (y - meany_ho)
  C /= (n - 2)
  return C
