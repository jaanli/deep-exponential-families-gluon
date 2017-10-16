import abc
import numpy as np
import mxnet as mx

from mxnet import nd
from mxnet import gluon
from typing import Union
from .distributions import BaseDistribution
from common import util

ZERO = nd.array([0.])
ONE = nd.array([1.])


class BaseBernoulli(BaseDistribution, metaclass=abc.ABCMeta):

  @property
  def is_reparam(self):
    return False

  mean = None
  logits = None

  def sample(self, n_samples: int = 1) -> nd.NDArray:
    mean = self.get_param_not_repeated('mean')
    if n_samples == 1:
      return nd.sample_uniform(ZERO, ONE, shape=mean.shape) < mean
    else:
      shape = (n_samples,) + mean.shape
      return nd.sample_uniform(ZERO, ONE, shape=shape)[0, :] < mean

  def log_prob(self, x: nd.NDArray) -> nd.NDArray:
    logits = self.get_param_maybe_repeated('logits')
    if x.ndim > logits.ndim:
      logits = nd.expand_dims(logits, 0)
    return x * logits - util.softplus(logits)


class Bernoulli(BaseBernoulli):
  def __init__(self, logits: nd.NDArray) -> None:
    super(Bernoulli, self).__init__()
    self.logits = logits

  @property
  def mean(self):
    return util.sigmoid(self.logits)


class FastBernoulli(BaseBernoulli):
  """Fast parameterization of Bernoulli as in the survival filter paper.

  Complexity O(CK) + O(CK) reduced to O(CK) + O(sK) where s in nonzeros.

  References:
  http://auai.org/uai2015/proceedings/papers/246.pdf
  """

  def __init__(self,
               positive_latent: nd.NDArray,
               weight: nd.NDArray,
               bias: nd.NDArray) -> None:
    """Number of classes is C; latent dimension K.

    Args:
      positive_latent: shape [batch_size, K] positive latent variable
      weight: shape [K, C] real-valued weight
    """
    super(FastBernoulli, self).__init__()
    # mean_arg is of shape [batch_size, C]
    self._positive_latent = positive_latent
    self._weight = weight
    self._bias = bias
    self.logits = None

  @property
  def mean(self):
    arg = nd.dot(self._positive_latent, nd.exp(
        self._weight)) + nd.exp(self._bias)
    return 1. - nd.exp(-arg)

  def log_prob(self, nonzero_index):
    raise NotImplementedError("Not implemented!")

  def log_prob_sum(self, nonzero_index: nd.NDArray) -> nd.NDArray:
    """Returns log prob. Argument is batch of indices of nonzero classes.
    log p(x) = term_1 + term_2
    term_1 = sum_c log p(x_c = 0)
    term_2 = sum_{c: x_c = 1} log p(x_c = 1) - log p(x_c = 0)
    term_1 takes O(CK) to calculate.
    term_2 takes O(CK) + O(sK) with s being the number of nonzero entries in x
    """
    mean_arg = -(nd.dot(self._positive_latent, nd.exp(self._weight))
                 + nd.exp(self._bias))
    assert mean_arg.shape[1] == 1, "Fast Bernoulli only supports batch size 1!"
    mean_arg = mean_arg[:, 0, :]
    term_1 = nd.sum(mean_arg, -1)
    n_factors, n_classes = self._weight.shape
    # weight_nonzero = nd.Embedding(
    #     nonzero_index, self._weight.T, n_classes, n_factors).T
    # nonzero_arg = -nd.dot(self._positive_latent, nd.exp(weight_nonzero))
    # raise NotImplementedError('need to add bias lookup!')
    batch_size = mean_arg.shape[0]
    nonzero_arg = nd.Embedding(
        nonzero_index, mean_arg.T, n_classes, batch_size).T
    term_2 = nd.sum(nd.log(1. - nd.exp(nonzero_arg)) - nonzero_arg, -1)
    res = term_1 + term_2
    return nd.expand_dims(res, 1)
