import os
import mxnet as mx
import numpy as np
import typing
import gensim
import scipy.sparse
import sklearn.decomposition
import sklearn.metrics.pairwise
import logging
logger = logging.getLogger(__name__)

from mxnet import nd
from mxnet import gluon


def print_top_words(weight: gluon.Parameter,
                    id2word: dict,
                    top: int = 10) -> None:
  n_factors, vocab_size = weight.shape
  weight = weight.data().asnumpy()
  for factor_idx in range(n_factors):
    top_word_indices = np.argsort(weight[factor_idx])[::-1][0:top]
    logger.info('----------')
    logger.info('factor %d:' % factor_idx)
    for word_idx in top_word_indices:
      logger.info('%.3e\t%s' %
                  (weight[factor_idx, word_idx], id2word[word_idx]))


def tokenize_text(fname: str,
                  vocab_size: int,
                  invalid_label: int = -1,
                  start_label: int = 0) -> typing.Tuple[list, dict]:
  """Get tokenized sentences and vocab."""
  if not os.path.isfile(fname):
    raise IOError('Data file is not a file! Got: %s' % fname)
  lines = open(fname).readlines()
  lines = [line.rstrip('\n') for line in lines]
  # lines = [filter(None, i.split(' ')) for i in lines]
  lines = [i.split(' ') for i in lines]
  vocab = gensim.corpora.dictionary.Dictionary(lines)
  vocab.filter_extremes(no_below=0, no_above=1, keep_n=vocab_size)
  vocab = {v: k for k, v in vocab.items()}
  lines = [[w for w in sent if w in vocab] for sent in lines]
  sentences, vocab = mx.rnn.encode_sentences(
      lines, vocab=vocab, invalid_label=invalid_label, start_label=start_label)
  sentences = [sent for sent in sentences if len(sent) > 0]
  return sentences, vocab


def flatten(l): return [item for sublist in l for item in sublist]


def principal_components(sentences: list,
                         vocab_size: int,
                         n_components: int) -> list:
  """PCA on list of integers."""
  # sparse format
  row_ind = flatten(
      [[i] * len(sentence) for i, sentence in enumerate(sentences)])
  col_ind = flatten(sentences)
  shape = (len(sentences), vocab_size)
  data = np.ones(len(col_ind))
  X = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
  X_std = X - X.mean(axis=0)
  X_std = X_std / (1e-8 + np.std(X_std, 0))
  Y = sklearn.decomposition.PCA(n_components=n_components).fit_transform(X_std)
  return Y


def print_nearest_cosine_distance(embeddings: gluon.Parameter,
                                  id2word: dict,
                                  num: int = 10) -> None:
  embeddings = embeddings.data().asnumpy().T
  top_wordids = list(id2word.keys())[0:num]
  distances = sklearn.metrics.pairwise.cosine_similarity(
      embeddings[top_wordids], embeddings)
  for idx, distance in zip(top_wordids, distances):
    top_word_indices = np.argsort(distance)[::-1][1:11]
    logger.info('----------')
    logger.info("nearest words in cosine distance to: %s" % id2word[idx])
    for nearest in top_word_indices:
      logger.info('%.3e\t%s' % (distance[nearest], id2word[nearest]))


def _batchify_sentences(data: list,
                        vocab_size: int) -> typing.Tuple[nd.sparse.CSRNDArray, nd.NDArray]:
  """Collate data, a list of sentence, label tuples into a sparse batch."""
  indptr = [0]  # row offsets
  indices = []
  labels = []
  for row_idx, sentence_and_label in enumerate(data):
    sentence, label = sentence_and_label
    ptr = indptr[row_idx] + len(sentence)
    indptr.append(ptr)
    indices.extend(sentence)
    labels.append(label)
  values = [1] * len(indices)
  labels = nd.array(labels)
  batch = nd.sparse.csr_matrix(data=values,
                               indices=indices,
                               indptr=indptr,
                               shape=(len(data), vocab_size))
  return batch, labels


class SentenceDataLoader(gluon.data.DataLoader):
  def __init__(self,
               id2word: dict,
               principal_components: np.array=None,
               data_distribution: str='bernoulli',
               **kwargs) -> None:
    super(SentenceDataLoader, self).__init__(**kwargs)
    self.id2word = id2word
    self.principal_components = principal_components
    self.batch_size = kwargs['batch_size']
    self.data_distribution = data_distribution
    if data_distribution == 'indices' and self.batch_size != 1:
      raise ValueError(
          "Need batch size of 1 for variable-length index representation!")
    self.vocab_size = len(id2word)

  def __iter__(self):
    for batch in self._batch_sampler:
      if self.data_distribution == 'bernoulli':
        yield _batchify_sentences(
            [self._dataset[idx] for idx in batch], self.vocab_size)
      elif self.data_distribution == 'fast_bernoulli':
        res = [self._dataset[idx] for idx in batch]
        assert len(res) == 1
        data, label = res[0]
        yield nd.array(data), nd.array([label])


def _batchify_documents(data: list,
                        vocab_size: int) -> typing.Tuple[nd.sparse.CSRNDArray, nd.NDArray]:
  """Collate data, a list of sentence, label tuples into a sparse batch."""
  indptr = [0]  # row offsets
  indices = []
  labels = []
  values = []
  for row_idx, doc_and_label in enumerate(data):
    doc, label = doc_and_label
    ptr = indptr[row_idx] + len(doc)
    indptr.append(ptr)
    word_ids, counts = zip(*doc)
    indices.extend(word_ids)
    values.extend(counts)
    labels.append(label)
  labels = nd.array(labels).astype(np.int64)
  batch = nd.sparse.csr_matrix(data=values,
                               indices=indices,
                               indptr=indptr,
                               shape=(len(data), vocab_size))
  return batch, labels


class DocumentDataLoader(gluon.data.DataLoader):
  def __init__(self,
               id2word: dict,
               **kwargs) -> None:
    super(DocumentDataLoader, self).__init__(**kwargs)
    self.id2word = id2word
    self.vocab_size = len(id2word)

  def __iter__(self):
    for batch in self._batch_sampler:
      yield _batchify_documents(
          [self._dataset[idx] for idx in batch], self.vocab_size)
