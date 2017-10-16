import numpy as np
import tester


def test_latent_layer():
  """Test matching q(z) to p(z) where p is Gaussian."""
  mean = 5.

  config = """
  n_iterations: 50
  learning_rate: 0.1
  gradient:
    estimator: pathwise
    n_samples: 1
    batch_size: 1
  layer_1:
    latent_distribution: gaussian
    size: 1
    p_z_mean: {}
    p_z_variance: 1.
  """.format(mean)

  def test_posterior_predictive(sample: np.array) -> None:
    print('posterior predictive mean:', sample)
    print('prior mean:', mean)
    np.testing.assert_allclose(sample, mean, rtol=1e-1)

  tester.test(config, data=np.array([np.nan]),
              test_fn=test_posterior_predictive)
