import os
import copy
import time
import mxnet as mx
import numpy as np
import deep_exp_fam
import logging
logger = logging.getLogger(__name__)


from mxnet import gluon
from mxnet import nd
from deep_exp_fam import DeepExponentialFamilyModel
from deep_exp_fam import layers
from common import util
from common import data


def fit(cfg: dict,
        train_data: gluon.data.DataLoader) -> DeepExponentialFamilyModel:
  """Fit a deep exponential family model to data."""
  mx.random.seed(32429)
  np.random.seed(423323)
  ctx = [mx.gpu(0)] if ('use_gpu' in cfg and cfg['use_gpu']) else [mx.cpu()]
  with mx.Context(ctx[0]):
    my_model = DeepExponentialFamilyModel(
        n_data=len(train_data._dataset), gradient_config=cfg['gradient'])
    with my_model.name_scope():
      layer_names = sorted([key for key in cfg if 'layer' in key])[::-1]
      for name in layer_names:
        if name != 'layer_0':
          my_model.add(layers.LatentLayer, **cfg[name])
        elif name == 'layer_0':
          my_model.add(layers.ObservationLayer, **cfg[name])
    params = my_model.collect_params()
    logger.info(params)
    latest_step = 0
    if 'dir' in cfg:
      latest_params, latest_step = util.latest_checkpoint(cfg['dir'])
      if latest_params is not None:
        my_model.load_params(latest_params, ctx)
    step = latest_step if latest_step is not None else 0
    params.initialize(ctx=ctx)
    my_model.maybe_attach_repeated_params()
    trainer = mx.gluon.Trainer(
        params, 'rmsprop', {'learning_rate': cfg['learning_rate']})
    print_every = cfg['print_every'] if 'print_every' in cfg else 100
    while step <= cfg['n_iterations'] + 1:
      for data_batch in train_data:
        with mx.autograd.record():
          log_q_sum, elbo, sample = my_model(data_batch, False)
          my_model.compute_gradients(elbo, data_batch, log_q_sum)
        if step % print_every == 0:
          elbo = np.mean(np.mean(elbo.asnumpy(), 0), 0)
          if 'layer_0' in cfg and hasattr(train_data, 'id2word'):
            w = params.get('observationlayer0_point_mass_weight')
            data.print_top_words(w, train_data.id2word)
          logger.info('t %d\telbo: %.3e' % (step, elbo))
          if np.isnan(elbo):
            raise ValueError('ELBO hit nan!')
          if 'dir' in cfg:
            param_file = 'my_model.params-iteration-%d' % step
            my_model.save_params(os.path.join(cfg['dir'], param_file))
        trainer.step(1)
        step += 1
        if step >= cfg['n_iterations']:
          break
  return my_model
