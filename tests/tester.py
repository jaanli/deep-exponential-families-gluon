import mxnet as mx
import numpy as np
import yaml

from common import fit
from mxnet import gluon
from mxnet import nd
from typing import Callable

import deep_exp_fam


def test(config_yaml: str, data: np.array, test_fn: Callable = None):
  np.random.seed(23423)

  def get_data_iter(batch_size, shuffle):
    dataset = gluon.data.ArrayDataset(
        data.astype(np.float32), range(len(data)))
    return gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  config = yaml.load(config_yaml)
  data_iter = get_data_iter(config['gradient']['batch_size'], shuffle=True)
  my_model = fit.fit(config, data_iter)
  data_iter = get_data_iter(config['gradient']['batch_size'], shuffle=False)
  n_samples_stats = 10
  for data_batch in data_iter:
    _, _, sample = my_model(data_batch)
    tmp_sample = nd.zeros_like(sample)
    for _ in range(n_samples_stats):
      _, _, sample = my_model(data_batch)
      tmp_sample += sample
    tmp_sample /= n_samples_stats
    if tmp_sample.ndim == 3:
      tmp_sample = nd.mean(tmp_sample, 0, keepdims=True)
    tmp_sample = tmp_sample.asnumpy()
    if len(data) == 1:
      tmp_sample = tmp_sample.reshape((1, -1))
      test_fn(tmp_sample)
    else:
      test_fn(tmp_sample, data_batch[0].asnumpy())
