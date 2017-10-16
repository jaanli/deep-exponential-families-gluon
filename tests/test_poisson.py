import numpy as np
import distributions
import mxnet as mx
import scipy.stats
import scipy.special

from mxnet import nd

mx.random.seed(13343)


def test_poisson_sampling():
  rate = 5.
  n_samples = 10000
  samples = distributions.Poisson(nd.array([rate])).sample(n_samples)
  mean = nd.mean(samples).asnumpy()
  np.testing.assert_allclose(mean, rate, rtol=1e-2)


def test_poisson_log_prob():
  rate = 1.
  data = [2, 5, 0, 10, 4]
  np_log_prob = scipy.stats.poisson.logpmf(np.array(data), mu=np.array(rate))
  p = distributions.Poisson(nd.array([rate]))
  mx_log_prob = p.log_prob(nd.array(data)).asnumpy()
  np.testing.assert_allclose(mx_log_prob, np_log_prob)
