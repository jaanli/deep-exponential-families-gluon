import numpy as np
import h5py
import yaml
import os
import time
import mxnet as mx

from mxnet import nd
from common import fit


if __name__ == '__main__':
  path = os.path.join(
      os.environ['LOG'], '/mnist/' + time.strftime("%Y-%m-%d"))
  if not os.path.exists(path):
    os.makedirs(path)

  config = yaml.load(
      """
      dir: {}
      clear_dir: false
      use_gpu: false
      learning_rate: 0.01
      n_iterations: 100000
      print_every: 1
      gradient:
        estimator: score_function
        n_samples: 32
        batch_size: 10
      layer_1:
        latent_distribution: poisson
        # weight_distribution: point_mass
        size: 10
      layer_0:
        weight_distribution: point_mass
        data_distribution: bernoulli
        data_size: 784
      """.format(path))

  if config['clear_dir']:
    for f in os.listdir(config['dir']):
      os.remove(os.path.join(config['dir'], f))

  # hdf5 file from:
  # https://github.com/altosaar/proximity_vi/blob/master/get_binary_mnist.py
  data_path = os.path.join(os.environ['DAT'], 'binarized_mnist.hdf5')

  f = h5py.File(data_path, 'r')
  raw_data = f['train'][:][0:10]
  f.close()

  train_data = mx.io.NDArrayIter(
      data={'data': nd.array(raw_data)},
      label={'label': range(len(raw_data)) * np.ones((len(raw_data),))},
      batch_size=config['gradient']['batch_size'],
      shuffle=True)

  fit.fit(config, train_data)
