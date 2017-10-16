import numpy as np
"""Run Gibbs sampling in the model:

  z ~ N(0, 1)
  w ~ N(0, 1)
  x ~ N(zw, 1)
"""
np.random.seed(1323)


def sample_latent(mu_a, var_a, b_sample, mu_b, var_b, x, var_x):
  """Sample a latent variable w or z."""
  mean = ((1. / var_b * x * b_sample + 1. / var_a * mu_a) /
          (1. / var_x * np.square(b_sample) + 1. / var_a))
  var = 1. / (1. / var_x * np.square(b_sample) + 1. / var_a)
  return mean + var * np.random.normal()


def normal_log_prob(x, mean, var):
  return -0.5 * np.log(2. * np.pi * var) - 0.5 * np.square(x - mean) / var


def run_gibbs_sampling():
  mu_z = 0.
  var_z = 1.
  mu_w = 0.
  var_w = 1.
  x = 30.3
  var_x = 1.

  w_sample = mu_w + var_w * np.random.normal()
  for step in range(100):
    z_sample = sample_latent(mu_z, var_z, w_sample, mu_w, var_w, x, var_x)
    w_sample = sample_latent(mu_w, var_w, z_sample, mu_z, var_z, x, var_x)
    log_prob = (normal_log_prob(x, z_sample * w_sample, var_x) +
                normal_log_prob(w_sample, mu_w, var_w) +
                normal_log_prob(z_sample, mu_z, var_z))
    if step % 10 == 0:
      print('step:', step)
      print('log joint', log_prob)
      print('z_sample:', z_sample)
      print('w_sample:', w_sample)
      x_sample = z_sample * w_sample + var_x * np.random.normal()
      print('posterior predictive mean:', z_sample * w_sample)


if __name__ == '__main__':
  run_gibbs_sampling()
