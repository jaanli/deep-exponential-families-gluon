import numpy as np
import distributions
import scipy.stats
import mxnet as mx


mx.random.seed(13343)


def test_bernoulli_sampling():
  logits = 0.232
  n_samples = 10000
  p = distributions.Bernoulli(mx.nd.array([logits]))
  samples = p.sample(n_samples)
  mean = mx.nd.mean(samples).asnumpy()
  print('sampling mean, mean', mean, p.mean.asnumpy())
  np.testing.assert_allclose(mean, p.mean.asnumpy(), rtol=1e-2)


def test_bernoulli_log_prob():
  logits = 0.384
  data = [0, 1, 0, 0, 1]
  p = distributions.Bernoulli(mx.nd.array([logits]))
  np_log_prob = scipy.stats.bernoulli.logpmf(
      np.array(data), p=p.mean.asnumpy())
  mx_log_prob = p.log_prob(mx.nd.array(data)).asnumpy()
  np.testing.assert_allclose(mx_log_prob, np_log_prob)
