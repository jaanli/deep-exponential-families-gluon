import numpy as np
import gensim
import yaml
import os
import time
from mxnet import nd
from mxnet import gluon
from common import fit
from common import util
from common import data


if __name__ == '__main__':
  path = os.path.join(os.environ['LOG'], 'text/' + time.strftime("%Y-%m-%d"))

  if not os.path.exists(path):
    os.makedirs(path)

  cfg = yaml.load(
      """
      dir: {}
      clear_dir: false
      # IMPORTANT
      use_gpu: true
      learning_rate: 0.01
      n_iterations: 5000000
      print_every: 10000
      gradient:
        estimator: score_function
        n_samples: 32
        batch_size: 64
      layer_1:
        latent_distribution: poisson
        q_z_mean: 3.
        size: 100
      layer_0:
        weight_distribution: point_mass
        data_distribution: poisson
        data_size: null
      """.format(path))

  if cfg['clear_dir']:
    for f in os.listdir(cfg['dir']):
      os.remove(os.path.join(cfg['dir'], f))

  with open(os.path.join(cfg['dir'], 'config.yml'), 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

  util.log_to_file(os.path.join(cfg['dir'], 'train.log'))

  fname = os.path.join(os.environ['DAT'], 'science/documents_train.dat')
  fname_vocab = os.path.join(os.environ['DAT'], 'science/VOCAB-TFIDF-1000.dat')

  corpus = gensim.corpora.bleicorpus.BleiCorpus(fname, fname_vocab)
  cfg['layer_0']['data_size'] = len(corpus.id2word)
  docs = [doc for doc in corpus if len(doc) > 0]
  dataset = gluon.data.ArrayDataset(data=docs, label=range(len(docs)))
  train_data = data.DocumentDataLoader(dataset=dataset,
                                       id2word=corpus.id2word,
                                       batch_size=cfg['gradient']['batch_size'],
                                       last_batch='discard',
                                       shuffle=True)

  fit.fit(cfg, train_data)
