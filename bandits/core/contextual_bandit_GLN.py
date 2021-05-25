#from absl import app
#from absl import flags

import haiku as hk
import jax
import jax.numpy as jnp
import rlax

#from gated_linear_networks import bernoulli
#from gated_linear_networks.examples import utils
#import bernoulli
#from examples import utils

MAX_TRAIN_STEPS = 2000

# Small example network, achieves ~95% test set accuracy =======================
# Network parameters.
NUM_LAYERS = 2

NEURONS_PER_LAYER = 70

CONTEXT_DIM = 1


# Learning rate schedule.
MAX_LR = 0.003

LR_CONSTANT = 1.0

LR_DECAY = 0.1


# Logging parameters.
EVALUATE_EVERY = 1000


def main(unused_argv):

  # Build a (binary) GLN classifier ============================================
  def network_factory():

    def gln_factory():
      output_sizes = [NEURONS_PER_LAYER] * NUM_LAYERS + [1]
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

  @jax.jit
  def accuracy(params, state, image, label):
    """One-vs-all classifier inference fn."""
    fn = jax.vmap(inference_, in_axes=(0, 0, None))
    predictions, unused_state = fn(params, state, image)
    return (jnp.argmax(predictions) == label).astype(jnp.float32)

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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def run_contextual_bandit(context_dim, num_actions, dataset, algos):
  """Run a contextual bandit problem on a set of algorithms.

  Args:
    context_dim: Dimension of the context.
    num_actions: Number of available actions.
    dataset: Matrix where every row is a context + num_actions rewards.
    algos: List of algorithms to use in the contextual bandit instance.

  Returns:
    h_actions: Matrix with actions: size (num_context, num_algorithms).
    h_rewards: Matrix with rewards: size (num_context, num_algorithms).
  """

  num_contexts = dataset.shape[0]

  # Create contextual bandit
  cmab = ContextualBandit(context_dim, num_actions)
  cmab.feed_data(dataset)

  h_actions = np.empty((0, len(algos)), float)
  h_rewards = np.empty((0, len(algos)), float)
  h_accumulative_rewards = np.zeros((1, len(algos)), float)
  h_regrets = np.zeros((1, len(algos)), float)


  # Run the contextual bandit process
  for i in range(num_contexts):
    context = cmab.context(i)
    actions = [a.action(context) for a in algos]
    actions = []
    rewards = [cmab.reward(i, action) for action in actions]
    opt_reward = cmab.reward(i, cmab.optimal(i))

    for j, a in enumerate(algos):
      a.update(context, actions[j], rewards[j])

    h_actions = np.vstack((h_actions, np.array(actions)))
    h_rewards = np.vstack((h_rewards, np.array(rewards)))
    h_accumulative_rewards = np.vstack((h_accumulative_rewards, h_accumulative_rewards[-1]+np.array(rewards)))
    h_regrets = np.vstack((h_regrets, h_accumulative_rewards[-1] - opt_reward))

  return h_actions, h_rewards, h_accumulative_rewards, h_regrets


class ContextualBandit(object):
  """Implements a Contextual Bandit with d-dimensional contexts and k arms."""

  def __init__(self, context_dim, num_actions):
    """Creates a contextual bandit object.

    Args:
      context_dim: Dimension of the contexts.
      num_actions: Number of arms for the multi-armed bandit.
    """

    self._context_dim = context_dim
    self._num_actions = num_actions

  def feed_data(self, data):
    """Feeds the data (contexts + rewards) to the bandit object.

    Args:
      data: Numpy array with shape [n, d+k], where n is the number of contexts,
        d is the dimension of each context, and k the number of arms (rewards).

    Raises:
      ValueError: when data dimensions do not correspond to the object values.
    """

    if data.shape[1] != self.context_dim + self.num_actions:
      raise ValueError('Data dimensions do not match.')

    self._number_contexts = data.shape[0]
    self.data = data
    print(f"!!!!{data.shape}")
    print(f"!!!{self.context_dim}")
    self.order = range(self.number_contexts)

  def reset(self):
    """Randomly shuffle the order of the contexts to deliver."""
    self.order = np.random.permutation(self.number_contexts)

  def context(self, number):
    """Returns the number-th context."""
    return self.data[self.order[number]][:self.context_dim]

  def reward(self, number, action):
    """Returns the reward for the number-th context and action."""
    return self.data[self.order[number]][self.context_dim + action]

  def optimal(self, number):
    """Returns the optimal action (in hindsight) for the number-th context."""
    return np.argmax(self.data[self.order[number]][self.context_dim:])

  @property
  def context_dim(self):
    return self._context_dim

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def number_contexts(self):
    return self._number_contexts
