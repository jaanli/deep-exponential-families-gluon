import numpy as np
import time
import distributions
import scipy.stats
import scipy.special
import mxnet as mx
from mxnet import nd


mx.random.seed(13343)
np.random.seed(2324)


def test_bernoulli_sampling():
  n_samples = 10000
  K = 10  # num factors
  C = 2  # num classes
  # latent variable is of size [n_samples, batch_size, latent_size]
  positive_latent = nd.ones((1, 1, K)) * 0.01
  weight = nd.ones((K, C)) * 0.1
  bias = nd.ones(C) * 0.01
  p = distributions.FastBernoulli(
      positive_latent=positive_latent, weight=weight, bias=bias)
  samples = p.sample(n_samples)
  print(samples.shape)
  mean = nd.mean(samples, 0).asnumpy()
  print('sampling mean, mean', mean, p.mean.asnumpy())
  np.testing.assert_allclose(mean, p.mean.asnumpy(), rtol=1e-1)


def test_bernoulli_log_prob():
  K = 10  # num factors
  C = 100  # num classes
  positive_latent = nd.ones((1, 1, K)) * nd.array(np.random.rand(K))
  weight = nd.ones((K, C)) * nd.array(np.random.rand(K, C))
  bias = nd.ones(C) * 0.01
  data = np.random.binomial(n=1, p=0.1, size=C)
  assert np.sum(data) > 0
  nonzero_idx = np.nonzero(data)[0]
  p = distributions.FastBernoulli(
      positive_latent=positive_latent, weight=weight, bias=bias)
  np_log_prob_sum = scipy.stats.bernoulli.logpmf(
      np.array(data), p=p.mean.asnumpy()).sum()
  mx_log_prob_sum = p.log_prob_sum(
      nonzero_index=nd.array(nonzero_idx)).asnumpy()
  print('mx log prob sum, np log prob sum', mx_log_prob_sum, np_log_prob_sum)
  np.testing.assert_allclose(mx_log_prob_sum, np_log_prob_sum, rtol=1e-3)
