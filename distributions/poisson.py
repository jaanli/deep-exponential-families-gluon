import abc
import numpy as np
import mxnet as mx

from mxnet import nd
from mxnet import gluon
from typing import Union
from typing import Callable
from .distributions import BaseDistribution
from common import util


class BasePoisson(BaseDistribution, metaclass=abc.ABCMeta):

  @property
  def is_reparam(self):
    return False

  mean = None

  def sample(self, n_samples: int = 1) -> nd.NDArray:
    mean = self.get_param_not_repeated('mean')
    if n_samples == 1:
      res = nd.sample_poisson(mean)
    else:
      res = nd.sample_poisson(mean, n_samples)
      res = nd.transpose(res)
    if res.ndim == 3:
      return nd.swapaxes(res, 1, 2)
    elif res.ndim == 2:
      return res
    else:
      raise ValueError('Ambiguous sample shape.')

  def log_prob(self, x: nd.NDArray) -> nd.NDArray:
    mean = self.get_param_maybe_repeated('mean')
    if x.ndim > mean.ndim:
      mean = nd.expand_dims(mean, 0)
    np_x = x.asnumpy().astype(np.int32).astype(np.float32)
    np.testing.assert_almost_equal(x.asnumpy(), np_x)
    return x * nd.log(mean) - mean - nd.gammaln(x + 1.)


class Poisson(BasePoisson):
  def __init__(self, mean: nd.NDArray) -> None:
    super(Poisson, self).__init__()
    self._mean = mean

  @property
  def mean(self):
    return self._mean


class PriorPoisson(BasePoisson):
  def __init__(self,
               name: str,
               shape: Union[int, tuple],
               mean: float = 1.) -> None:
    super(PriorPoisson, self).__init__()
    with self.name_scope():
      assert mean > 0.
      self.mean = self.params.get(name,
                                  shape=shape,
                                  init=mx.init.Constant(mean),
                                  grad_req='null')

  def __call__(self,
               z_above: nd.NDArray,
               weight: nd.NDArray,
               bias: nd.NDArray) -> Poisson:
    """Call the prior layer at a lower layer of a DEF."""
    mean = util.softplus(nd.dot(z_above, weight) + bias)
    return Poisson(mean)


class VariationalPoisson(BasePoisson):
  def __init__(self, name: str, shape: tuple, mean: float = 1.) -> None:
    super(VariationalPoisson, self).__init__()
    with self.name_scope():
      mean_arg_init = mx.init.Constant(util.np_inv_softplus(mean))
      self.mean = self.params.get(
          name + '_mean_arg', init=mean_arg_init, shape=shape)
      self.mean.link_function = util.softplus


class VariationalLookupPoisson(BasePoisson):
  def __init__(self,
               name: str,
               shape: tuple,
               mean: float = 1.,
               init_pca: np.array = None) -> None:
    """Mean-field factorized variational Poisson. Per-datapoint parameters."""
    super(VariationalLookupPoisson, self).__init__()
    with self.name_scope():
      assert mean > 0.
      if init_pca is not None:
        # assert np.all(init_pca > 0)
        pca_mean = np.mean(util.np_softplus(init_pca), -1)
        correction = util.np_inverse_softplus(mean - pca_mean)
        init_pca_arg = init_pca + np.expand_dims(correction, -1)
        mean_arg_init = mx.init.Constant(init_pca_arg)
      else:
        mean_arg_init = UniformInit(mean=mean,
                                    scale=0.07,
                                    inverse_transform=util.np_inverse_softplus)
      self._mean_arg_emb = self.params.get(
          name + '_mean_arg_emb', init=mean_arg_init, shape=shape)
      self.link_function = util.softplus

  def lookup(self, labels: nd.NDArray, repeat: bool =True):
    """Return the distribution for the data batch."""
    shape = self._mean_arg_emb.shape
    mean_arg_emb = nd.Embedding(labels, self._mean_arg_emb.data(), *shape)
    self.mean = nd.maximum(5e-3, self.link_function(mean_arg_emb))
    if hasattr(self._mean_arg_emb, 'n_repeats') and repeat:
      self.mean_repeated = self.link_function(
          util.repeat_emb(self._mean_arg_emb, mean_arg_emb))
    return self


@mx.init.register
class UniformInit(mx.init.Initializer):
  def __init__(self,
               mean: float = 0.,
               scale: float = 0.07,
               inverse_transform: Callable = None) -> None:
    super(UniformInit, self).__init__()
    if inverse_transform is not None:
      self.low = inverse_transform(mean - scale)
      self.high = inverse_transform(mean + scale)
    else:
      self.low = mean - scale
      self.high = mean + scale

  def _init_weight(self, _, arr):
    arr[:] = nd.random.uniform(low=self.low, high=self.high, shape=arr.shape)

  def _init_bias(self, _, arr):
    arr[:] = nd.zeros(shape=arr.shape)
