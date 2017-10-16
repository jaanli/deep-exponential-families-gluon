import pdb
import numpy as np
import mxnet as mx
assert mx.__version__ > '0.11.0'  # required for autograd.grad()

from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from typing import Tuple

from .layers import LatentLayer
from .layers import ObservationLayer

EPSILON = np.finfo(float).eps


class DeepExponentialFamilyModel(object):
  def __init__(self, gradient_config: dict, n_data: int) -> None:
    self.n_data = n_data
    self.gradient_config = gradient_config
    self.sequential = gluon.nn.Sequential()
    self.size_above = None
    self._grads_attached = False

  def name_scope(self):
    return self.sequential.name_scope()

  def add(self, block, **kwargs):
    if block == LatentLayer:
      block = block(n_data=self.n_data,
                    size_above=self.size_above,
                    gradient_config=self.gradient_config,
                    **kwargs)
      self.sequential.add(block)
      self.size_above = kwargs['size']
    elif block == ObservationLayer:
      self.sequential.add(
          block(size_above=self.size_above,
                gradient_config=self.gradient_config,
                **kwargs))
    else:
      raise ValueError('Unknown block type: %s' % type(block))

  def maybe_attach_repeated_params(self):
    """Attach repeated params if using score function estimator gradient."""
    cfg = self.gradient_config
    params = self.collect_params()
    point_params = [p for name, p in params.items() if 'point' in name]
    self._point_mass_params = point_params
    if cfg['estimator'] == 'pathwise':
      self._grads_attached = True
      return
    elif cfg['estimator'] == 'score_function':
      for name, param in params.items():
        if 'point_mass' not in name and param.grad_req != 'null':
          assert cfg['n_samples'] >= 3, "Require n_samples >=3 for gradient."
          if 'emb' in name:
            param.n_repeats = cfg['n_samples']
            # this seems to be slower than the dense version!
            # param._data[0] = param._data[0].tostype('row_sparse')
            # autograd.mark_variables(
            #     param._data[0], nd.zeros(param.shape).tostype('row_sparse'))
          else:
            res = nd.repeat(param.data(), repeats=cfg['batch_size'], axis=0)
            res = nd.expand_dims(res, 0)
            param.repeated = nd.repeat(res, repeats=cfg['n_samples'], axis=0)
            # request gradient with respect to each sample and batch datapoint
            param.repeated.attach_grad()
          param.score_grad = True
      score_params = [p for p in params.values() if hasattr(p, 'score_grad')]
      self._score_params = score_params
      self._params = params
      self._grads_attached = True

  def collect_params(self):
    return self.sequential.collect_params()

  def save_params(self, *args):
    self.sequential.save_params(*args)

  def load_params(self, *args):
    self.sequential.load_params(*args)

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def forward(self,
              data_batch: mx.io.DataBatch,
              get_sample: bool = True,
              ) -> Tuple[nd.NDArray, nd.NDArray, nd.NDArray]:
    """Return ELBO, reconstruction or z_sample from the DEF."""
    if not self._grads_attached:
      raise ValueError('Must call maybe_attach_repeated_params() first!')
    log_q_sum, elbo, sample = None, None, None
    for i, block in enumerate(self.sequential._children):
      log_q_sum, elbo, sample = block(data_batch, log_q_sum, elbo, sample)
    return log_q_sum, elbo, sample

  def compute_gradients(self,
                        elbo: nd.NDArray,
                        data_batch: mx.io.DataBatch = None,
                        log_q_sum: nd.NDArray = None,
                        mode: str = 'train') -> None:
    """Compute gradients and assign them to variational parameters.

    Args:
      elbo: evidence lower bound that we maximize
      data_batch: minibatch of data with data indices as labels
      log_q_sum: sum of log probs of samples from variational distributions q.
    """
    cfg = self.gradient_config
    if cfg['estimator'] == 'pathwise':
      for block in self.sequential._children:
        for child_block in block._children:
          if hasattr(child_block, 'is_reparam'):
            assert child_block.is_reparam == True
    if len(self._point_mass_params) > 0 and mode == 'train':
      variables = [p.data() for p in self._point_mass_params]
      assert elbo.shape[-1] == cfg['batch_size']
      loss = nd.mean(-elbo, -1)
      point_mass_grads = autograd.grad(loss, variables, retain_graph=True)
      _assign_grads(self._point_mass_params, point_mass_grads)
    if cfg['estimator'] == 'pathwise':
        (-elbo).backward()
    elif cfg['estimator'] == 'score_function':
      variables = [param.repeated for param in self._score_params]
      score_functions = autograd.grad(log_q_sum, variables)
      mx.autograd.set_recording(False)
      score_grads = []
      for param, score_function in zip(self._score_params, score_functions):
        grad = _leave_one_out_gradient_estimator(score_function, -elbo)
        if 'emb' in param.name:
          # turns out the sparse implementation is not faster?!
          # data, label = data_batch
          # label = label.astype(np.int64)
          # grad = nd.sparse.row_sparse_array(
          #     grad, indices=label, shape=param.shape)
          # need to broadcast for embeddings
          one_hot = nd.one_hot(data_batch[1], depth=self.n_data)
          grad = nd.dot(one_hot, grad, transpose_a=True)
        score_grads.append(grad)
      _assign_grads(self._score_params, score_grads)


def _leave_one_out_gradient_estimator(h, f, zero_mean_h=False):
  """Estimate gradient of f using score function and control variate h.

  Optimal scaling of control variate is given by: a = Cov(h, f) / Var(h).
  """
  if h.ndim > f.ndim:
    # expand parameter dimension (last dimension summed over in f)
    f = nd.expand_dims(f, f.ndim)
  grad_f = h * f
  if zero_mean_h:
    cov_h_f = _leave_one_out_mean(h * grad_f)
    var_h = _leave_one_out_mean(h * h)
  else:
    cov_h_f = _held_out_covariance(h, grad_f)
    var_h = _held_out_covariance(h, h)
  # sampling zero for low-variance score functions is probable, so add EPSILON!
  optimal_a = cov_h_f / (EPSILON + var_h)
  if h.ndim == 2:
    # If no batch dim: nd.Embedding removes batch dim for batches of size 1
    keepdims = True
  else:
    keepdims = False
  return nd.mean(grad_f - optimal_a * h, 0, keepdims=keepdims)


def _leave_one_out_mean(a: nd.NDArray) -> nd.NDArray:
  """Compute leave-one-out mean of array of shape [n_samples, ...]."""
  n_samples = a.shape[0]
  assert n_samples >= 256, "Need at least 256 samples for accuracy."
  res = (nd.sum(a, 0, keepdims=True) - a) / (n_samples - 2)
  assert res.shape == a.shape
  return res


def _held_out_covariance(x, y):
  """Get held-out covariance between x and y in the first dimension."""
  n = x.shape[0]
  assert y.shape[0] == n
  mean_x = nd.mean(x, 0)
  mean_y = nd.mean(y, 0)
  res = nd.sum((x - mean_x) * (y - mean_y), 0)
  mean_x_held_out = (mean_x - x / n) / (1. - 1. / n)  # * n / (n - 1.)
  mean_y_held_out = (mean_y - y / n) / (1. - 1. / n)  # * n / (n - 1.)
  res = res - (x - mean_x_held_out) * (y - mean_y_held_out)
  return res / (n - 2.)


def _variance(a: nd.NDArray) -> nd.NDArray:
  """Compute variance of a of shape [n_samples, ...]."""
  mean = nd.mean(a, 0, keepdims=True)
  return nd.mean(nd.square(a - mean), 0)


def _assign_grads(params: list, grads: list):
  """Assign gradients in the context for the parameter."""
  for param, grad in zip(params, grads):
    assert param._grad[0].shape == grad.shape
    param._grad[0] = grad
    param._data[0]._fresh_grad = 1
