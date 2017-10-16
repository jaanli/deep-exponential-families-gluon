import mxnet as mx
import distributions

from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from typing import Tuple
from typing import Callable
from common import util


def _filter_kwargs(prefix, kwargs):
  """Return new kwargs matching a prefix, or empty dict if no matches."""
  tmp_kwargs = {}
  for kwarg, value in kwargs.items():
    if prefix in kwarg:
      kwarg = kwarg.lstrip(prefix)
      kwarg = kwarg.lstrip('_')
      tmp_kwargs[kwarg] = value
  return tmp_kwargs


def _build_distribution(
        name: str,
        distribution_type: str,
        shape: tuple,
        **kwargs) -> distributions.BaseDistribution:
  kwargs = _filter_kwargs(name, kwargs)
  if name.startswith('p_'):
    attr = 'Prior'
  elif name.startswith('q_'):
    attr = 'Variational'
  if attr == 'Variational' and name.endswith('_z'):
    # assume mean-field variational inference with per-datapoint parameters
    attr += 'Lookup'
  attr += distribution_type.capitalize()
  dist_class = getattr(distributions, attr)
  return dist_class(name, shape, **kwargs)


class LatentLayer(gluon.Block):
  def __init__(self,
               latent_distribution: str,
               n_data: int,
               size: int,
               size_above: int,
               gradient_config: dict,
               weight_distribution: str = None,
               **kwargs) -> None:
    super(LatentLayer, self).__init__()
    self.gradient_config = gradient_config
    with self.name_scope():
      self.size_above = size_above
      self.p_z = _build_distribution(
          'p_z', latent_distribution, (1, size), **kwargs)
      self.q_z = _build_distribution(
          'q_z', latent_distribution, (n_data, size), **kwargs)
      if size_above is not None:
        weight_shape = (size_above, size)
        if weight_distribution == 'point_mass':
          self.params.get('point_mass_weight', shape=weight_shape)
          self.params.get('point_mass_bias', shape=(
              size,), init=mx.init.Zero())
        else:
          raise NotImplementedError(
              'Need to implement non point-mass weight distribution!')

  def forward(self,
              data: tuple,
              log_q_sum: nd.NDArray,
              elbo_above: nd.NDArray,
              z_above: nd.NDArray,
              ) -> Tuple[nd.NDArray, nd.NDArray, nd.NDArray]:
    n_samples = self.gradient_config['n_samples']
    q_z = self.q_z.lookup(data[1])
    z_sample = q_z.sample(n_samples)
    if self.gradient_config['estimator'] == 'score_function':
      # important -- do not differentiate through latent sample
      z_sample = nd.stop_gradient(z_sample)
    log_q_z = nd.sum(q_z.log_prob(z_sample), -1)
    elbo = elbo_above if elbo_above is not None else 0.
    log_q_sum = log_q_sum if log_q_sum is not None else 0.
    log_q_sum = log_q_sum + log_q_z
    if self.size_above is None:
      # first (top) layer, without weights
      log_p_z = nd.sum(self.p_z.log_prob(z_sample), -1)
      elbo = elbo + log_p_z - log_q_z
    elif self.size_above is not None:
      w = self.params.get('point_mass_weight').data()
      b = self.params.get('point_mass_bias').data()
      log_p_z = nd.sum(self.p_z(z_above, w, b).log_prob(z_sample), -1)
      elbo = elbo + log_p_z - log_q_z
    assert log_q_sum.shape[0] == n_samples
    assert elbo.shape[0] == n_samples
    return log_q_sum, elbo, z_sample


class ObservationLayer(gluon.Block):
  def __init__(self,
               weight_distribution: str,
               size_above: int,
               data_size: int,
               gradient_config: dict,
               data_distribution: str = 'gaussian',
               **kwargs) -> None:
    super(ObservationLayer, self).__init__()
    self.gradient_config = gradient_config
    self.data_distribution = data_distribution
    with self.name_scope():
      weight_shape = (size_above, data_size)
      if weight_distribution == 'point_mass':
        # w_init = UniformInit(mean=-4.)
        self.params.get('point_mass_weight', shape=weight_shape)
        self.params.get(
            'point_mass_bias', shape=(data_size,), init=mx.init.Zero())
      else:
        raise NotImplementedError(
            'Weights and biases other than point mass not implemented!')

  def p_x_fn(self,
             z_above: nd.NDArray,
             weight: nd.NDArray,
             bias: nd.NDArray = None) -> distributions.BaseDistribution:
    # z_above: [n_samples, batch_size, size_above]
    # weight: [size_above, data_size]
    if self.data_distribution == 'gaussian':
      params = nd.dot(z_above, weight) + bias
      variance = nd.ones_like(params)
      return distributions.Gaussian(params, variance)
    elif self.data_distribution == 'bernoulli':
      params = nd.dot(z_above, weight) + bias
      return distributions.Bernoulli(logits=params)
    elif self.data_distribution == 'poisson':
      # minimum intercept is 0.01
      return distributions.Poisson(
          0.01 + nd.dot(z_above, util.softplus(weight)))
    else:
      raise ValueError(
          'Incompatible data distribution: %s' % self.data_distribution)

  def forward(self,
              data: tuple,
              log_q_sum: nd.NDArray,
              elbo_above: nd.NDArray,
              z_above: nd.NDArray) -> Tuple[nd.NDArray, nd.NDArray, nd.NDArray]:
    n_samples = self.gradient_config['n_samples']
    w = self.params.get('point_mass_weight').data()
    b = self.params.get('point_mass_bias').data()
    p_x = self.p_x_fn(z_above, w, b)
    log_p_x = nd.sum(p_x.log_prob(data[0]), -1)
    elbo = elbo_above + log_p_x
    return log_q_sum, elbo, p_x.mean
