import numpy as np
import time


def normal_log_prob(x, mean, var):
  return -0.5 * np.log(2. * np.pi * var) - 0.5 * np.square(x - mean) / var


def grad_mean_normal_log_prob(x, mean, var):
  return (x - mean) / var


def grad_var_normal_log_prob(x, mean, var):
  return -0.5 / var + 0.5 * np.square(x - mean) / np.square(var)


def softplus(x):
  return np.log(1. + np.exp(x))


def grad_softplus(x):
  return np.exp(x) / (1. + np.exp(x))


def inv_softplus(x):
  return np.log(np.exp(x) - 1.)


def grad_normal_log_prob(x, mean, var_arg):
  grad_var_arg = (grad_var_normal_log_prob(x, mean, softplus(var_arg))
                  * grad_softplus(var_arg))
  grad_mean = grad_mean_normal_log_prob(x, mean, softplus(var_arg))
  return grad_mean, grad_var_arg


def normal_sample(mean, var_arg, n_samples):
  return mean + np.sqrt(softplus(var_arg)) * np.random.normal(size=n_samples)


def bbvi():
  p_z_mean = 0.
  p_z_var = 1.
  q_z_mean = 0.
  q_z_var_arg = inv_softplus(1.)
  q_w_mean = 0.
  q_w_var_arg = inv_softplus(1.)
  p_w_mean = 0.
  p_w_var = 1.
  p_x_var = 1.
  x = 30.
  n_samples = 32
  learning_rate = 1e-5

  for i in range(500000):
    t0 = time.time()
    z_sample = normal_sample(q_z_mean, q_z_var_arg, n_samples)
    log_q_z = normal_log_prob(z_sample, q_z_mean, softplus(q_z_var_arg))
    log_p_z = normal_log_prob(z_sample, p_z_mean, p_z_var)
    w_sample = normal_sample(q_w_mean, q_w_var_arg, n_samples)
    log_q_w = normal_log_prob(w_sample, q_w_mean, softplus(q_w_var_arg))
    log_p_w = normal_log_prob(w_sample, p_w_mean, p_w_var)
    p_x_mean = w_sample * z_sample
    log_p_x = normal_log_prob(x, p_x_mean, p_x_var)
    elbo = log_p_x + log_p_z - log_q_z + log_p_w - log_q_w

    # gradients
    score_q_z_mean, score_q_z_var_arg = grad_normal_log_prob(
        z_sample, q_z_mean, q_z_var_arg)
    score_q_w_mean, score_q_w_var_arg = grad_normal_log_prob(
        w_sample, q_w_mean, q_w_var_arg)

    if i % 1000 == 0:
      print('i: %d\t elbo: %.3f\tposterior predicitve mean, data: %.3f, %.3f' %
            (i, np.mean(elbo), np.mean(p_x_mean), x))
      print('q_w_mean: %.3f\tq_z_mean: %.3f' % (q_w_mean, q_z_mean))
      print('q_w_var: %.3f\tq_z_var: %.3f' %
            (softplus(q_w_var_arg), softplus(q_z_var_arg)))

    # updates
    q_z_mean += learning_rate * bbvi_gradient(score_q_z_mean, elbo)
    q_z_var_arg += learning_rate * bbvi_gradient(score_q_z_var_arg, elbo)
    q_w_mean += learning_rate * bbvi_gradient(score_q_w_mean, elbo)
    q_w_var_arg += learning_rate * bbvi_gradient(score_q_w_var_arg, elbo)
    t1 = time.time()
    # print('time: ', t1 - t0)


def bbvi_gradient(score_function, elbo):
  grad_elbo = score_function * elbo
  cov = leave_one_out_mean(score_function * grad_elbo)
  var = leave_one_out_mean(score_function * score_function)
  a = cov / var
  return np.mean(grad_elbo - score_function * a)


def leave_one_out_mean(a):
  return (np.sum(a, 0, keepdims=True) - a) / (a.shape[0] - 1.)


if __name__ == '__main__':
  bbvi()
