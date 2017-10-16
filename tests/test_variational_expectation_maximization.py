import numpy as np
import tester


def test_one_layer_gaussian():
  """Test variational EM with pathwise gradients."""
  config = """
  learning_rate: 0.1
  n_iterations: 100
  gradient:
    estimator: pathwise
    n_samples: 1
    batch_size: 1
  layer_1:
    latent_distribution: gaussian
    p_z_variance: 1.
    size: 1
  layer_0:
    weight_distribution: point_mass
    p_w_variance: 1.
    data_size: 1
  """
  data = np.array([[3.3]])

  def test_posterior_predictive(sample: np.array) -> None:
    print('data:', data)
    print('posterior predictive sample:', sample)
    np.testing.assert_allclose(sample, data, rtol=0.3)

  tester.test(config, data=data, test_fn=test_posterior_predictive)


def test_one_layer_gaussian_score():
  """Test variational EM with score function gradients."""
  config = """
  learning_rate: 0.1
  n_iterations: 100
  print_every: 100
  gradient:
    estimator: score_function
    n_samples: 32
    batch_size: 1
  layer_1:
    latent_distribution: gaussian
    size: 1
  layer_0:
    weight_distribution: point_mass
    data_size: 1
  """
  data = np.array([[30.3]])

  def test_posterior_predictive(sample: np.array) -> None:
    print('data:', data)
    print('posterior predictive sample:', sample)
    np.testing.assert_allclose(sample, data, rtol=1e-1)

  tester.test(config, data=data, test_fn=test_posterior_predictive)


def test_one_layer_poisson_score():
  """Test variational EM with score function gradients and poisson latents."""
  config = """
  learning_rate: 0.1
  n_iterations: 100
  print_every: 100
  gradient:
    estimator: score_function
    n_samples: 32
    batch_size: 1
  layer_1:
    latent_distribution: poisson
    size: 1
  layer_0:
    weight_distribution: point_mass
    data_size: 1
  """
  data = np.array([[30.3]])

  def test_posterior_predictive(sample: np.array) -> None:
    print('data:', data)
    print('posterior predictive sample:', sample)
    np.testing.assert_allclose(sample, data, rtol=0.3)

  tester.test(config, data=data, test_fn=test_posterior_predictive)


def test_two_layer_gaussian_score():
  """Test variational EM with score function gradients and poisson latents."""
  config = """
  learning_rate: 0.1
  n_iterations: 100
  gradient:
    estimator: score_function
    n_samples: 32
    batch_size: 1
  layer_2:
    latent_distribution: gaussian
    size: 1
  layer_1:
    latent_distribution: gaussian
    weight_distribution: point_mass
    size: 1
  layer_0:
    weight_distribution: point_mass
    data_size: 1
  """
  data = np.array([[8.5]])

  def test_posterior_predictive(sample: np.array) -> None:
    print('data:', data)
    print('posterior predictive sample:', sample)
    np.testing.assert_allclose(sample, data, rtol=1e-1)

  tester.test(config, data=data, test_fn=test_posterior_predictive)
