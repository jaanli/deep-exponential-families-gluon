import os
import logging
import mxnet as mx
import numpy as np
import collections

from mxnet import nd


def log_to_file(filename):
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                      datefmt='%m-%d %H:%M',
                      filename=filename,
                      filemode='a')
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)


def flatten(l): return [item for sublist in l for item in sublist]


def softplus(x):
  return nd.Activation(x, act_type='softrelu')


def np_softplus(x):
  return np.log(1. + np.exp(x))


class Softplus(object):
  def __init__(self):
    pass

  def __call__(self, x: nd.NDArray) -> nd.NDArray:
    return nd.Activation(x, act_type='softrelu')

  def backward(self, x):
    # d/dx log(1 + exp(x)) = exp(x) / (1 + exp(x)) = 1. / (1. + exp(-x))
    return nd.sigmoid(x)


def sigmoid(x):
  return nd.Activation(x, act_type='sigmoid')


def np_inverse_softplus(x):
  return np.log(np.exp(x) - 1.)


def latest_checkpoint(directory):
  files = [f for f in os.listdir(directory) if 'params' in f]
  if len(files) > 0 and any('params' in f for f in files):
    l = sorted((int(f.split('-')[-1]), i) for i, f in enumerate(files))
    return os.path.join(directory, files[l[-1][-1]]), l[-1][0]
  else:
    return None, None


def repeat_emb(param, emb):
  """Maybe repeat an embedding."""
  res = nd.expand_dims(emb, 0)
  param.repeated = nd.repeat(res, repeats=param.n_repeats, axis=0)
  param.repeated.attach_grad()
  return param.repeated


def pathwise_grad_variance_callback(my_model, data_batch):
  """Get pathwise gradient estimator variance."""
  param_grads = collections.defaultdict(lambda: [])  # type: ignore
  params = my_model.collect_params()
  n_samples_stats = 10
  for i in range(n_samples_stats):
    with mx.autograd.record():
      log_q_sum, elbo, sample = my_model(data_batch)
      my_model.compute_gradients(elbo, data_batch, log_q_sum)
      for name, param in params.items():
        if param.grad_req != 'null':
          param_grads[name].append(param.grad().asnumpy())


def callback_elbo_sample(my_model, data_batch):
  """Get a reduced-variance estimate of the elbo and sample."""
  n_samples_stats = 10
  _, elbo, sample = my_model(data_batch)
  for _ in range(n_samples_stats):
    tmp_sample = nd.zeros_like(sample)
    tmp_elbo = nd.zeros_like(elbo)
    for _ in range(n_samples_stats):
      _, elbo, sample = my_model(data_batch)
      tmp_sample += sample
      tmp_elbo += elbo
    tmp_sample /= n_samples_stats
    tmp_elbo /= n_samples_stats
    tmp_sample = np.mean(tmp_sample.asnumpy(), 0)
    tmp_elbo = np.mean(tmp_elbo.asnumpy())
  return tmp_elbo, tmp_sample


def score_grad_variance_callback(my_model):
  """Get score function gradient variance."""
  params = my_model.collect_params()
  param_grads = collections.defaultdict(lambda: [])  # type: ignore
  for name, param in params.items():
    if param.grad_req != 'null':
      grads = np.stack(param_grads[name])
      param.grad_variance = np.mean(np.var(grads, axis=0))
      param.grad_norm = np.mean(np.linalg.norm(grads, axis=-1))
      for block in my_model.sequential._children:
        print(block.name, ':')
        print([(name, p.data().asnumpy().tolist())
               for name, p in filter(
            lambda x: 'weight' in x[0] or 'bias' in x[0],
            block.collect_params().items())])
        for child_block in block._children:
          print(child_block.name, ':')
          print('mean:', child_block.get_param_not_repeated('mean').asnumpy())
          if hasattr(child_block, 'variance'):
            print('variance: ', child_block.get_param_not_repeated(
                'variance').asnumpy())
