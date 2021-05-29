from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Haiku modules for feature processing."""

import copy
from typing import Tuple

import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import interpolation
import tensorflow_datasets as tfds

Array = chex.Array


def _moments(image):
  """Compute the first and second moments of a given image."""
  c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]
  total_image = np.sum(image)
  m0 = np.sum(c0 * image) / total_image
  m1 = np.sum(c1 * image) / total_image
  m00 = np.sum((c0 - m0)**2 * image) / total_image
  m11 = np.sum((c1 - m1)**2 * image) / total_image
  m01 = np.sum((c0 - m0) * (c1 - m1) * image) / total_image
  mu_vector = np.array([m0, m1])
  covariance_matrix = np.array([[m00, m01], [m01, m11]])
  return mu_vector, covariance_matrix


def _deskew(image):
  """Image deskew."""
  c, v = _moments(image)
  alpha = v[0, 1] / v[0, 0]
  affine = np.array([[1, 0], [alpha, 1]])
  ocenter = np.array(image.shape) / 2.0
  offset = c - np.dot(affine, ocenter)
  return interpolation.affine_transform(image, affine, offset=offset)


def _deskew_dataset(dataset):
  """Dataset deskew."""
  deskewed = copy.deepcopy(dataset)
  for k, before in dataset.items():
    images = before["image"]
    num_images = images.shape[0]
    after = np.stack([_deskew(i) for i in np.squeeze(images, axis=-1)], axis=0)
    deskewed[k]["image"] = np.reshape(after, (num_images, -1))
  return deskewed


def load_deskewed_mnist(*a, **k):
  """Returns deskewed MNIST numpy dataset."""
  mnist_data, info = tfds.load(*a, **k)
  mnist_data = tfds.as_numpy(mnist_data)
  deskewed_data = _deskew_dataset(mnist_data)
  return deskewed_data, info


class MeanStdEstimator(hk.Module):
  """Online mean and standard deviation estimator using Welford's algorithm."""

  def __call__(self, sample: jnp.DeviceArray) -> Tuple[Array, Array]:
    if len(sample.shape) > 1:
      raise ValueError("sample must be a rank 0 or 1 DeviceArray.")

    count = hk.get_state("count", shape=(), dtype=jnp.int32, init=jnp.zeros)
    mean = hk.get_state(
        "mean", shape=sample.shape, dtype=jnp.float32, init=jnp.zeros)
    m2 = hk.get_state(
        "m2", shape=sample.shape, dtype=jnp.float32, init=jnp.zeros)

    count += 1
    delta = sample - mean
    mean += delta / count
    delta_2 = sample - mean
    m2 += delta * delta_2

    hk.set_state("count", count)
    hk.set_state("mean", mean)
    hk.set_state("m2", m2)

    stddev = jnp.sqrt(m2 / count)
    return mean, stddev

"""Bernoulli Gated Linear Network."""

from typing import List, Text, Tuple

import chex
import jax
import jax.numpy as jnp
import rlax
import tensorflow_probability as tfp
import inspect
from gated_linear_networks import base

#tfp = tfp.experimental.substrates.jax
#tfd = tfp.distributions

Array = chex.Array

GLN_EPS = 0.05
MAX_WEIGHT = 200.


class GatedLinearNetwork(base.GatedLinearNetwork):
  """Bernoulli Gated Linear Network."""

  def __init__(self,
               output_sizes: List[int],
               context_dim: int,
               name: Text = "bernoulli_gln"):
    """Initialize a Bernoulli GLN."""
    super(GatedLinearNetwork, self).__init__(
        output_sizes,
        context_dim,
        inference_fn=GatedLinearNetwork._inference_fn,
        update_fn=GatedLinearNetwork._update_fn,
        init=jnp.zeros,
        dtype=jnp.float32,
        name=name)

  def _add_bias(self, inputs):
    #print("??????")
    #print(inspect.stack()[1])
    #print(f'_add_bias: {inputs}')
    #print(f'?? input type: {type(inputs)}')
    #print(f'??? tuple len: {len(inputs)}')
    #print(inputs[0].shape)
    #print(inputs[1].shape)
    #print(f'?? input shape: {inputs.shape}')
    return jnp.append(inputs, rlax.sigmoid(1.))
    #return 0

  @staticmethod
  def _inference_fn(
      inputs: Array,           # [input_size]
      side_info: Array,        # [side_info_size]
      weights: Array,          # [2**context_dim, input_size]
      hyperplanes: Array,      # [context_dim, side_info_size]
      hyperplane_bias: Array,  # [context_dim]
  ) -> Array:
    """Inference step for a single Beurnolli neuron."""

    weight_index = GatedLinearNetwork._compute_context(side_info, hyperplanes,
                                                       hyperplane_bias)
    #print(f'weight_index: {weight_index}')
    used_weights = weights[weight_index]
    inputs = rlax.logit(jnp.clip(inputs, GLN_EPS, 1. - GLN_EPS))
    prediction = rlax.sigmoid(jnp.dot(used_weights, inputs))

    return prediction, weight_index

  @staticmethod
  def _update_fn(
      inputs: Array,           # [input_size]
      side_info: Array,        # [side_info_size]
      weights: Array,          # [2**context_dim, num_features]
      hyperplanes: Array,      # [context_dim, side_info_size]
      hyperplane_bias: Array,  # [context_dim]
      target: Array,           # []
      learning_rate: float,
  ) -> Tuple[Array, Array, Array]:
    """Update step for a single Bernoulli neuron."""

    def log_loss_fn(inputs, side_info, weights, hyperplanes, hyperplane_bias,
                    target):
      """Log loss for a single Bernoulli neuron."""
      prediction, _ = GatedLinearNetwork._inference_fn(inputs, side_info, weights,
                                                    hyperplanes,
                                                    hyperplane_bias)
      prediction = jnp.clip(prediction, GLN_EPS, 1. - GLN_EPS)
      return rlax.log_loss(prediction, target), prediction

    grad_log_loss = jax.value_and_grad(log_loss_fn, argnums=2, has_aux=True)
    ((log_loss, prediction),
     dloss_dweights) = grad_log_loss(inputs, side_info, weights, hyperplanes,
                                     hyperplane_bias, target)

    delta_weights = learning_rate * dloss_dweights
    new_weights = jnp.clip(weights - delta_weights, -MAX_WEIGHT, MAX_WEIGHT)
    return new_weights, prediction, log_loss


class LastNeuronAggregator(base.LastNeuronAggregator):
  """Bernoulli last neuron aggregator, implemented by the super class."""
  pass



import time
from absl import app
from absl import flags
import numpy as np
import os
import tensorflow as tf
from hparams import HParams
from matplotlib import pyplot as plt

from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import sample_adult_data
from bandits.data.data_sampler import sample_census_data
from bandits.data.data_sampler import sample_covertype_data
from bandits.data.data_sampler import sample_jester_data
from bandits.data.data_sampler import sample_mushroom_data
from bandits.data.data_sampler import sample_statlog_data
from bandits.data.data_sampler import sample_stock_data
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_sparse_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
from bandits.algorithms.uniform_sampling import UniformSampling

import haiku as hk
import jax
import jax.numpy as jnp
import rlax
import math

#from gated_linear_networks import bernoulli
#from gated_linear_networks.examples import utils
#import bernoulli
#from examples import utils

MAX_TRAIN_STEPS = 2000

# Small example network, achieves ~95% test set accuracy =======================
# Network parameters.
#NUM_LAYERS = 2

#NEURONS_PER_LAYER = 70

CONTEXT_DIM = 3
S = 8
EXPLORATION_C = 0.03
NETWORK_SHAPE = [100, 10, 1]

# Learning rate schedule.
MAX_LR = 0.1

LR_CONSTANT = 1.0

LR_DECAY = 0.1


# Logging parameters.
#EVALUATE_EVERY = 1000
EVALUATE_EVERY = 1

# Set up your file routes to the data files.
base_route = os.getcwd()
data_route = 'contextual_bandits/datasets'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '/tmp/bandits/', 'Base directory to save output')
flags.DEFINE_string(
    'mushroom_data',
    os.path.join(base_route, data_route, 'mushroom.data'),
    'Directory where Mushroom data is stored.')
flags.DEFINE_string(
    'financial_data',
    os.path.join(base_route, data_route, 'raw_stock_contexts'),
    'Directory where Financial data is stored.')
flags.DEFINE_string(
    'jester_data',
    os.path.join(base_route, data_route, 'jester_data_40jokes_19181users.npy'),
    'Directory where Jester data is stored.')
flags.DEFINE_string(
    'statlog_data',
    os.path.join(base_route, data_route, 'shuttle.trn'),
    'Directory where Statlog data is stored.')
flags.DEFINE_string(
    'adult_data',
    os.path.join(base_route, data_route, 'adult.full'),
    'Directory where Adult data is stored.')
flags.DEFINE_string(
    'covertype_data',
    os.path.join(base_route, data_route, 'covtype.data'),
    'Directory where Covertype data is stored.')
flags.DEFINE_string(
    'census_data',
    os.path.join(base_route, data_route, 'USCensus1990.data.txt'),
    'Directory where Census data is stored.')


def sample_data(data_type, num_contexts=None):
  """Sample data from given 'data_type'.

  Args:
    data_type: Dataset from which to sample.
    num_contexts: Number of contexts to sample.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
    opt_rewards: Vector of expected optimal reward for each context.
    opt_actions: Vector of optimal action for each context.
    num_actions: Number of available actions.
    context_dim: Dimension of each context.
  """

  if data_type == 'linear':
    # Create linear dataset
    num_actions = 8
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=noise_stds)
    opt_rewards, opt_actions = opt_linear
  elif data_type == 'sparse_linear':
    # Create sparse linear dataset
    num_actions = 7
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    num_nnz_dims = int(context_dim / 3.0)
    dataset, _, opt_sparse_linear = sample_sparse_linear_data(
        num_contexts, context_dim, num_actions, num_nnz_dims, sigma=noise_stds)
    opt_rewards, opt_actions = opt_sparse_linear
  elif data_type == 'mushroom':
    # Create mushroom dataset
    num_actions = 2
    context_dim = 117
    file_name = FLAGS.mushroom_data
    dataset, opt_mushroom = sample_mushroom_data(file_name, num_contexts)
    opt_rewards, opt_actions = opt_mushroom
  elif data_type == 'financial':
    num_actions = 8
    context_dim = 21
    num_contexts = min(3713, num_contexts)
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    file_name = FLAGS.financial_data
    dataset, opt_financial = sample_stock_data(file_name, context_dim,
                                               num_actions, num_contexts,
                                               noise_stds, shuffle_rows=True)
    opt_rewards, opt_actions = opt_financial
  elif data_type == 'jester':
    num_actions = 8
    context_dim = 32
    num_contexts = min(19181, num_contexts)
    file_name = FLAGS.jester_data
    dataset, opt_jester = sample_jester_data(file_name, context_dim,
                                             num_actions, num_contexts,
                                             shuffle_rows=True,
                                             shuffle_cols=True)
    opt_rewards, opt_actions = opt_jester
  elif data_type == 'statlog':
    file_name = FLAGS.statlog_data
    num_actions = 7
    num_contexts = min(43500, num_contexts)
    sampled_vals = sample_statlog_data(file_name, num_contexts,
                                       shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'adult':
    file_name = FLAGS.adult_data
    num_actions = 14
    num_contexts = min(45222, num_contexts)
    sampled_vals = sample_adult_data(file_name, num_contexts,
                                     shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'covertype':
    file_name = FLAGS.covertype_data
    num_actions = 7
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_covertype_data(file_name, num_contexts,
                                         shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'census':
    file_name = FLAGS.census_data
    num_actions = 9
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_census_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'wheel':
    delta = 0.95
    num_actions = 5
    context_dim = 2
    mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
    std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
    mu_large = 50
    std_large = 0.01
    dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta,
                                                  mean_v, std_v,
                                                  mu_large, std_large)
    opt_rewards, opt_actions = opt_wheel

  return dataset, opt_rewards, opt_actions, num_actions, context_dim


def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed after {} seconds.'.format(
    name, time.time() - t_init))
  print('---------------------------------------------------')

  performance_pairs = []
  for j, a in enumerate(algos):
    performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
  performance_pairs = sorted(performance_pairs,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')


def main(_):

  # Problem parameters
  num_contexts = 45000
  num_classes = 14
  # Data type in {linear, sparse_linear, mushroom, financial, jester,
  #                 statlog, adult, covertype, census, wheel}
  data_type = 'statlog'

  # Create dataset
  sampled_vals = sample_data(data_type, num_contexts)
  dataset, opt_rewards, opt_actions, num_actions, context_dim = sampled_vals
  # dataset: (num_contexts, context_dim + num_actions)
  # opt_rewards: (num_contexts,)
  dataset = dataset[:, :context_dim]
  labels = opt_actions

  # Build a (binary) GLN classifier ============================================
  def network_factory():

    def gln_factory():
      #output_sizes = [NEURONS_PER_LAYER] * NUM_LAYERS + [1]
      output_sizes = NETWORK_SHAPE
      return GatedLinearNetwork(
          output_sizes=output_sizes, context_dim=CONTEXT_DIM)

    return LastNeuronAggregator(gln_factory)

  def extract_features(image):
    mean, stddev = MeanStdEstimator()(image)
    standardized_img = (image - mean) / (stddev + 1.)
    inputs = rlax.sigmoid(standardized_img)
    side_info = standardized_img
    return inputs, side_info

  def inference_fn(image, *args, **kwargs):
    inputs, side_info = extract_features(image)
    return network_factory().inference(inputs, side_info, *args, **kwargs)

  def update_fn(image, *args, **kwargs):
    inputs, side_info = extract_features(image)
    return network_factory().update(inputs, side_info, *args, **kwargs)

  init_, inference_ = hk.without_apply_rng(
      hk.transform_with_state(inference_fn))
  _, update_ = hk.without_apply_rng(hk.transform_with_state(update_fn))

  # Map along class dimension to create a one-vs-all classifier ================
  @jax.jit
  def init(dummy_image, key):
    """One-vs-all classifier init fn."""
    dummy_images = jnp.stack([dummy_image] * num_classes, axis=0)
    keys = jax.random.split(key, num_classes)
    return jax.vmap(init_, in_axes=(0, 0))(keys, dummy_images)

  #@jax.jit
  def accuracy(params, state, image, label,):
    """One-vs-all classifier inference fn."""
    fn = jax.vmap(inference_, in_axes=(0, 0, None))
    (predictions, weight_indexes), unused_state = fn(params, state, image)
    #print(f'predictions: {jnp.argmax(predictions)}')
    #print(f'label: {label}')
    return (jnp.argmax(predictions) == label).astype(jnp.float32)

  @jax.jit
  def GLN(params, state, image):
    fn = jax.vmap(inference_, in_axes=(0, 0, None))
    (predictions, weight_indexes), state = fn(params, state, image)
    return predictions, weight_indexes

  @jax.jit
  def update(params, state, step, image, label):
    """One-vs-all classifier update fn."""

    # Learning rate schedules.
    learning_rate = jnp.minimum(
        MAX_LR, LR_CONSTANT / (1. + LR_DECAY * step))

    # Update weights and report log-loss.
    targets = hk.one_hot(jnp.asarray(label), num_classes)

    fn = jax.vmap(update_, in_axes=(0, 0, None, 0, None))
    out = fn(params, state, image, targets, learning_rate)
    (params, unused_predictions, log_loss), state = out
    return (jnp.mean(log_loss), params), state

  # Train on train split =======================================================
  #dummy_image = train_images[0]
  dummy_image = dataset[0]
  params, state = init(dummy_image, jax.random.PRNGKey(42))
  total_rewards = 0.
  # GLCB
  rewards = []
  regrets = []
  U = sum(NETWORK_SHAPE)
  N_su_a = jnp.zeros((num_classes, S), dtype=int)
  #N_su_a = np.zeros((num_classes, S), dtype=int)

  #@jax.jit
  def compute_N(step, a, signature):
    N_max = jnp.amax(N_su_a[a])
    step_arr = jnp.array([step]*U)
    N_s_a = jnp.take(N_su_a[a], signature)
    nu = jnp.sum(jnp.multiply(jnp.power(step_arr, N_s_a)/N_max, N_s_a))
    de = jnp.sum(jnp.power(step_arr, N_s_a)/N_max, N_s_a)
    return nu, de

  for step, (image, label) in enumerate(zip(dataset, labels), 1):
    # image -> context x_t
    #print(step)
    psudo_count = jnp.zeros((num_classes,))
    predictions, signatures = GLN(params, state, image)
    for a in range(num_classes):
      #nu = 0.
      #de = 0.
      N_max = jnp.amax(N_su_a[a])
      signature = signatures[a]
      
      step_arr = jnp.array([step]*U)
      N_s_a = jnp.take(N_su_a[a], signature)
      #print(N_s_a)
      if N_max != 0:
        nu = jnp.sum(jnp.multiply(jnp.power(step_arr, N_s_a/N_max), N_s_a))
        de = jnp.sum(jnp.power(step_arr, N_s_a/N_max))
      else:
        nu = jnp.sum(N_s_a)
        de = U
      '''
      if N_max == 0:
        nu = jnp.sum(jnp.take(N_su_a[a], signature))
        de = U
      else:
        nu, de = compute_N(step-1, a, signature)
      '''
      psudo_count = jax.ops.index_update(psudo_count, a, nu / de)
      '''
      if math.isnan(psudo_count[a]):
        print('nan')
        print(nu)
        print(de)
        print(f'N_s_a: {N_s_a}')
        print(f'step_arr: {step_arr}')
        print(f'power: {jnp.power(step_arr, N_s_a)}')
        print(f'divide: {jnp.power(step_arr, N_s_a)/N_max}')
        print(f'multiply: {jnp.multiply(jnp.power(step_arr, N_s_a)/N_max, N_s_a)}')
      elif math.isinf(psudo_count[a]):
        print('inf')
        print(nu)
        print(de)
        print(N_s_a)
      '''
    #print(psudo_count)
    explorations = EXPLORATION_C * jnp.sqrt(math.log(step) / psudo_count)
    #print(f'psudo_count: {psudo_count}')
    #print(f'inside sqrt: {math.log(step) / psudo_count}')
    #print(f'explorations: {explorations}')
    #print(f'predictions: {predictions}')
    ucb = predictions + explorations
    #print(ucb.shape)
    action = jnp.argmax(jnp.nan_to_num(ucb))
    reward = int(label == action)
    #print(N_su_a)
    #print(reward)
    #print('#####################################')
    total_rewards += reward
    (unused_loss, params), state = update(
        params,
        state,
        step,
        image,
        label,
    )
    regrets.append(1-reward)
    rewards.append(reward)
    signature = signatures[action]
    #update_N(signature, action)
    #update_N = np.zeros(S, dtype=int)
    #N_s_action = jnp.take(N_su_a[action, signature])
    update_N = np.array(N_su_a[action])
    for u in range(U):
      s = signature[u]
      update_N[s] += 1
      #N_su_a[action][s] += 1
    N_su_a = jax.ops.index_update(N_su_a, action, update_N)

    if MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS:
      print(f'Optimal total rewards for {data_type}: {step}')
      print(f'Total rewards gained by GLCB: {total_rewards}')
      '''
      with open('GLCB_statlog_rewards.txt', 'w') as f:
        for item in rewards:
          f.write("%s\n" % item)
      with open('GLCB_statlog_regrets.txt', 'w') as f:
        for item in regrets:
          f.write("%s\n" % item)
      '''
      return

if __name__ == '__main__':
  app.run(main)