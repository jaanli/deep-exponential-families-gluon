import numpy as np
import tester


def test_gaussian():
  config = """
  learning_rate: 0.1
  n_iterations: 300
  gradient:
    estimator: pathwise
    n_samples: 1
    batch_size: 1
  layer_1:
    latent_distribution: gaussian
    size: 1
  layer_0:
    weight_distribution: point_mass
    data_size: 1
  """
  data = np.array([[20.3], [-30.3], [15.]])

  def test_posterior_predictive(sample: np.array, data: np.array) -> None:
    print('----')
    print('data:', data)
    print('posterior predictive sample:', sample)
    np.testing.assert_allclose(sample, np.expand_dims(data, 0), rtol=0.2)

  tester.test(config, data=data, test_fn=test_posterior_predictive)


def test_gaussian_score():
  config = """
  learning_rate: 0.05
  n_iterations: 100
  gradient:
    estimator: score_function
    n_samples: 16
    batch_size: 3
  layer_1:
    latent_distribution: gaussian
    size: 7
  layer_0:
    weight_distribution: point_mass
    data_size: 1
  """
  data = np.array([[20.3], [30.3], [15.]])

  def test_posterior_predictive(sample: np.array, data: np.array) -> None:
    print('----')
    print('data:', data)
    print('posterior predictive sample:', sample)
    np.testing.assert_allclose(sample, np.expand_dims(data, 0), rtol=0.2)

  tester.test(config, data=data, test_fn=test_posterior_predictive)


def test_gaussian_score_multivariate_data():
  config = """
  learning_rate: 0.1
  n_iterations: 100
  gradient:
    estimator: score_function
    n_samples: 16
    batch_size: 3
  layer_1:
    latent_distribution: gaussian
    size: 7
  layer_0:
    weight_distribution: point_mass
    data_size: 3
  """
  data = np.array(
      [[20.3, -30.3, 15.3], [-30.3, -40.4, 23.5], [15., -20.3, 28.9]])

  def test_posterior_predictive(sample: np.array, data: np.array) -> None:
    print('----')
    print('data:', data)
    print('posterior predictive sample:', sample)
    np.testing.assert_allclose(sample, np.expand_dims(data, 0), rtol=0.2)

  tester.test(config, data=data, test_fn=test_posterior_predictive)


def test_poisson_multivariate_data():
  config = """
  learning_rate: 0.05
  n_iterations: 200
  gradient:
    estimator: score_function
    n_samples: 64
    batch_size: 3
  layer_1:
    latent_distribution: poisson
    size: 3
  layer_0:
    weight_distribution: point_mass
    data_size: 2
  """
  data = np.array(
      [[-20.3, 30.3], [-30.3, 40.4], [-10.5, 20.3]])

  def test_posterior_predictive(sample: np.array, data: np.array) -> None:
    print('----')
    print('data:', data)
    print('posterior predictive sample:', sample)
    np.testing.assert_allclose(sample, np.expand_dims(data, 0), rtol=0.5)

  tester.test(config, data=data, test_fn=test_posterior_predictive)
