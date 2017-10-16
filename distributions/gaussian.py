import mxnet as mx
import numpy as np
import abc

from .distributions import BaseDistribution
from typing import Union
from typing import Tuple
from mxnet import gluon
from mxnet import nd
from common import util


class BaseGaussian(BaseDistribution, metaclass=abc.ABCMeta):

  @property
  def is_reparam(self) -> bool:
    return True

  mean = None
  variance = None

  def sample(self, n_samples: int = 1) -> nd.NDArray:
    # reparameterization / pathwise trick for backprop
    mean = self.get_param_not_repeated('mean')
    variance = self.get_param_not_repeated('variance')
    shape = (n_samples,) + mean.shape
    return mean + nd.sqrt(variance) * nd.random_normal(shape=shape)

  def log_prob(self, x: nd.NDArray) -> nd.NDArray:
    mean = self.get_param_maybe_repeated('mean')
    variance = self.get_param_maybe_repeated('variance')
    if x.ndim > mean.ndim:
      mean = nd.expand_dims(mean, 0)
      variance = nd.expand_dims(variance, 0)
    diff = x - mean
    self._saved_for_backward = [diff]
    return (-0.5 * nd.log(2. * np.pi * variance)
            - nd.square(diff) / 2. / variance)


class Gaussian(BaseGaussian):
  def __init__(self,
               mean: nd.NDArray,
               variance: nd.NDArray) -> None:
    super(Gaussian, self).__init__()
    self.mean = mean
    self.variance = variance


class PriorGaussian(BaseGaussian):
  def __init__(self,
               name: str,
               shape: Union[int, tuple],
               mean: float = 0.,
               variance: float = 1.) -> None:
    super(PriorGaussian, self).__init__()
    with self.name_scope():
      self.mean = self.params.get(name + '_mean',
                                  init=mx.init.Constant(mean),
                                  shape=shape,
                                  grad_req='null')
      self.variance = self.params.get(name + '_variance',
                                      init=mx.init.Constant(variance),
                                      shape=shape,
                                      grad_req='null')

  def __call__(self,
               z_above: nd.NDArray,
               weight: nd.NDArray,
               bias: nd.NDArray) -> Gaussian:
    """Call the prior layer at a lower layer of a DEF."""
    mean = nd.dot(z_above, weight) + bias
    variance = self.variance
    return Gaussian(mean, variance)


class VariationalGaussian(BaseGaussian):
  def __init__(self,
               name: str,
               shape: tuple,
               mean: float = 0.,
               variance: float = 1.,
               grad_req: str = 'write') -> None:
    super(VariationalGaussian, self).__init__()
    mean_init = mx.init.Constant(mean)
    with self.name_scope():
      self.mean = self.params.get(name + '_mean', init=mean_init, shape=shape,
                                  grad_req=grad_req)
      variance_arg_init = mx.init.Constant(util.np_inverse_softplus(variance))
      self.variance = self.params.get(name + '_variance_arg',
                                      init=variance_arg_init, shape=shape,
                                      grad_req=grad_req)
      self.variance.link_function = util.softplus


class VariationalLookupGaussian(BaseGaussian):
  def __init__(self,
               name: str,
               shape: tuple,
               mean: float = 0.,
               variance: float = 1.,
               grad_req: str = 'write') -> None:
    """Mean-field factorized variational Gaussian. Per-datapoint parameters."""
    super(VariationalLookupGaussian, self).__init__()
    with self.name_scope():
      mean_init = mx.init.Constant(mean)
      self._mean_emb = self.params.get(
          name + '_mean_emb', init=mean_init, shape=shape, grad_req=grad_req)
      variance_arg_init = mx.init.Constant(util.np_inverse_softplus(variance))
      self._variance_arg_emb = self.params.get(
          name + '_variance_arg_emb', init=variance_arg_init, shape=shape,
          grad_req=grad_req)
      self.link_function = util.Softplus()

  def lookup(self, data_index: nd.NDArray):
    """Return the distribution for the data batch."""
    shape = self._mean_emb.shape
    self.mean = nd.Embedding(data_index, self._mean_emb.data(), *shape)
    variance_arg = nd.Embedding(
        data_index, self._variance_arg_emb.data(), *shape)
    self.variance = self.link_function(variance_arg)
    if hasattr(self._mean_emb, 'n_repeats'):
      self.mean_repeated = util.repeat_emb(self._mean_emb, self.mean)
    if hasattr(self._variance_arg_emb, 'n_repeats'):
      self.variance_repeated = self.link_function(
          util.repeat_emb(self._variance_arg_emb, variance_arg))
    return self
