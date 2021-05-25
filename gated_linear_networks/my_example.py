from typing import List, Text, Tuple

import chex
import jax
import jax.numpy as jnp
import rlax
import tensorflow_probability as tfp

import base

tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

Array = chex.Array

GLN_EPS = 0.01
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
    return jnp.append(inputs, rlax.sigmoid(1.))

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
    used_weights = weights[weight_index]
    inputs = rlax.logit(jnp.clip(inputs, GLN_EPS, 1. - GLN_EPS))
    prediction = rlax.sigmoid(jnp.dot(used_weights, inputs))

    return prediction

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
      prediction = GatedLinearNetwork._inference_fn(inputs, side_info, weights,
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

NEURONS_PER_LAYER = 100

CONTEXT_DIM = 1


# Learning rate schedule.
MAX_LR = 0.003

LR_CONSTANT = 1.0

LR_DECAY = 0.1


# Logging parameters.
EVALUATE_EVERY = 1000


def main(unused_argv):
  # Load MNIST dataset =========================================================
  mnist_data, info = load_deskewed_mnist(
      name='mnist', batch_size=-1, with_info=True)
  num_classes = info.features['label'].num_classes

  (train_images, train_labels) = (mnist_data['train']['image'],
                                  mnist_data['train']['label'])

  (test_images, test_labels) = (mnist_data['test']['image'],
                                mnist_data['test']['label'])

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

  # Train on train split =======================================================
  dummy_image = train_images[0]
  params, state = init(dummy_image, jax.random.PRNGKey(42))

  for step, (image, label) in enumerate(zip(train_images, train_labels), 1):
    (unused_loss, params), state = update(
        params,
        state,
        step,
        image,
        label,
    )

    # Evaluate on test split ===================================================
    if not step % EVALUATE_EVERY:
      batch_accuracy = jax.vmap(accuracy, in_axes=(None, None, 0, 0))
      accuracies = batch_accuracy(params, state, test_images, test_labels)
      total_accuracy = float(jnp.mean(accuracies))

      # Report statistics.
      print({
          'step': step,
          'accuracy': float(total_accuracy),
      })

    if MAX_TRAIN_STEPS is not None and step >= MAX_TRAIN_STEPS:
      return



main(0)