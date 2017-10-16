import abc
import typing

from mxnet import nd
from mxnet import gluon


class BaseDistribution(gluon.Block, metaclass=abc.ABCMeta):
  def __init__(self):
    super(BaseDistribution, self).__init__()

  @abc.abstractproperty
  def is_reparam(self):
    pass

  @abc.abstractmethod
  def sample(self, n_samples: int) -> nd.NDArray:
    pass

  @abc.abstractmethod
  def log_prob(self, x: nd.NDArray) -> nd.NDArray:
    pass

  def get_param_not_repeated(self, name):
    """Return a parameter without any repetitions."""
    param = getattr(self, name)
    if isinstance(param, gluon.Parameter):
      res = param.data()
      if hasattr(param, 'link_function'):
        return param.link_function(res)
      else:
        return res
    else:
      return param

  def get_param_maybe_repeated(self, name):
    """Return repeated parameter if it has been repeated to get jacobians."""
    if hasattr(self, name + '_repeated'):
      param = getattr(self, name + '_repeated')
    else:
      param = getattr(self, name)

    if isinstance(param, nd.NDArray):
      return param
    elif isinstance(param, gluon.Parameter):
      if hasattr(param, 'repeated'):
        res = param.repeated
      else:
        res = param.data()

      if hasattr(param, 'link_function'):
        return param.link_function(res)
      else:
        return res
    else:
      raise ValueError('Parameter has invalid type: %s' % type(param))

  def forward(self):
    raise ValueError('Ambiguous: Need to call log_prob or sample!')
