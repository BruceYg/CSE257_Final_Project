# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple example of contextual bandits simulation.

Code corresponding to:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks
for Thompson Sampling, by Carlos Riquelme, George Tucker, and Jasper Snoek.
https://arxiv.org/abs/1802.09127
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from bandits.data.data_sampler import classification_to_bandit_problem

import tensorflow as tf
import pandas as pd
import numpy as np

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '/tmp/bandits/', 'Base directory to save output')

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

  # Create dataset
  data_path = 'ml-100k/u.data'
  user_path = 'ml-100k/u.user'
  movie_path = 'ml-100k/u.item'
  with tf.io.gfile.GFile(data_path, 'r') as f:
    df = pd.read_csv(f, sep='\t', header=None, na_values=['?']).dropna()

  df.drop(3, axis=1, inplace=True)
  num_contexts = 2000
  num_actions = 5
  df = df.sample(frac=1)
  df = df.iloc[:num_contexts, :]
  labels = df[2].astype('category').cat.codes.to_numpy()
  pairs = df.drop(2, axis=1).to_numpy()
  user_indexes = pairs[:,0] - 1
  movie_indexes = pairs[:,1] - 1

  with tf.io.gfile.GFile(user_path, 'r') as f:
    df_user = pd.read_csv(f, sep='|', header=None, na_values=['?']).dropna()
  df_user.drop(4, axis=1, inplace=True)
  cols_to_transform = [2,3]
  df_user = pd.get_dummies(df_user, columns=cols_to_transform).to_numpy()
  user_data = np.take(df_user, user_indexes, axis=0)[:, 1:]

  df_movie = pd.read_csv(movie_path, header=None, sep='|', encoding='latin-1')
  df_movie = df_movie.drop([1,2,3,4], axis=1).to_numpy()
  movie_data = np.take(df_movie, movie_indexes, axis=0)[:, 1:]

  contexts = np.hstack((user_data, movie_data))
  sampled_vals = classification_to_bandit_problem(contexts, labels, num_actions)
  contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
  dataset = np.hstack((contexts, rewards))
  context_dim = contexts.shape[1]

  # Define hyperparameters and algorithms
  hparams = HParams(num_actions=num_actions)

  hparams_linear = HParams(num_actions=num_actions,
                                               context_dim=context_dim,
                                               a0=6,
                                               b0=6,
                                               lambda_prior=0.25,
                                               initial_pulls=2)

  hparams_rms = HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[50],
                                            batch_size=512,
                                            activate_decay=True,
                                            initial_lr=0.1,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=1000,
                                            buffer_s=-1,
                                            initial_pulls=2,
                                            optimizer='RMS',
                                            reset_lr=True,
                                            lr_decay_rate=0.5,
                                            training_freq=50,
                                            training_epochs=100,
                                            p=0.95,
                                            q=3)

  hparams_dropout = HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=[50],
                                                batch_size=512,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                optimizer='RMS',
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=50,
                                                training_epochs=100,
                                                use_dropout=True,
                                                keep_prob=0.80)

  hparams_bbb = HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[50],
                                            batch_size=512,
                                            activate_decay=True,
                                            initial_lr=0.1,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=1000,
                                            buffer_s=-1,
                                            initial_pulls=2,
                                            optimizer='RMS',
                                            use_sigma_exp_transform=True,
                                            cleared_times_trained=10,
                                            initial_training_steps=100,
                                            noise_sigma=0.1,
                                            reset_lr=False,
                                            training_freq=50,
                                            training_epochs=100)

  hparams_nlinear = HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=[50],
                                                batch_size=512,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=1,
                                                training_freq_network=50,
                                                training_epochs=100,
                                                a0=6,
                                                b0=6,
                                                lambda_prior=0.25)

  hparams_nlinear2 = HParams(num_actions=num_actions,
                                                 context_dim=context_dim,
                                                 init_scale=0.3,
                                                 activation=tf.nn.relu,
                                                 layer_sizes=[50],
                                                 batch_size=512,
                                                 activate_decay=True,
                                                 initial_lr=0.1,
                                                 max_grad_norm=5.0,
                                                 show_training=False,
                                                 freq_summary=1000,
                                                 buffer_s=-1,
                                                 initial_pulls=2,
                                                 reset_lr=True,
                                                 lr_decay_rate=0.5,
                                                 training_freq=10,
                                                 training_freq_network=50,
                                                 training_epochs=100,
                                                 a0=6,
                                                 b0=6,
                                                 lambda_prior=0.25)

  hparams_pnoise = HParams(num_actions=num_actions,
                                               context_dim=context_dim,
                                               init_scale=0.3,
                                               activation=tf.nn.relu,
                                               layer_sizes=[50],
                                               batch_size=512,
                                               activate_decay=True,
                                               initial_lr=0.1,
                                               max_grad_norm=5.0,
                                               show_training=False,
                                               freq_summary=1000,
                                               buffer_s=-1,
                                               initial_pulls=2,
                                               optimizer='RMS',
                                               reset_lr=True,
                                               lr_decay_rate=0.5,
                                               training_freq=50,
                                               training_epochs=100,
                                               noise_std=0.05,
                                               eps=0.1,
                                               d_samples=300,
                                              )

  hparams_alpha_div = HParams(num_actions=num_actions,
                                                  context_dim=context_dim,
                                                  init_scale=0.3,
                                                  activation=tf.nn.relu,
                                                  layer_sizes=[50],
                                                  batch_size=512,
                                                  activate_decay=True,
                                                  initial_lr=0.1,
                                                  max_grad_norm=5.0,
                                                  show_training=False,
                                                  freq_summary=1000,
                                                  buffer_s=-1,
                                                  initial_pulls=2,
                                                  optimizer='RMS',
                                                  use_sigma_exp_transform=True,
                                                  cleared_times_trained=10,
                                                  initial_training_steps=100,
                                                  noise_sigma=0.1,
                                                  reset_lr=False,
                                                  training_freq=50,
                                                  training_epochs=100,
                                                  alpha=1.0,
                                                  k=20,
                                                  prior_variance=0.1)

  hparams_gp = HParams(num_actions=num_actions,
                                           num_outputs=num_actions,
                                           context_dim=context_dim,
                                           reset_lr=False,
                                           learn_embeddings=True,
                                           max_num_points=1000,
                                           show_training=False,
                                           freq_summary=1000,
                                           batch_size=512,
                                           keep_fixed_after_max_obs=True,
                                           training_freq=50,
                                           initial_pulls=2,
                                           training_epochs=100,
                                           lr=0.01,
                                           buffer_s=-1,
                                           initial_lr=0.001,
                                           lr_decay_rate=0.0,
                                           optimizer='RMS',
                                           task_latent_dim=5,
                                           activate_decay=False)

  algos = [
      #UniformSampling('Uniform Sampling', hparams),
      #UniformSampling('Uniform Sampling 2', hparams),
      #FixedPolicySampling('fixed1', [0.75, 0.25], hparams),
      #FixedPolicySampling('fixed2', [0.25, 0.75], hparams),
      PosteriorBNNSampling('RMS', hparams_rms, 'RMSProp'),
      PosteriorBNNSampling('Dropout', hparams_dropout, 'RMSProp'),
      PosteriorBNNSampling('BBB', hparams_bbb, 'Variational'),
      NeuralLinearPosteriorSampling('NeuralLinear', hparams_nlinear),
      #NeuralLinearPosteriorSampling('NeuralLinear2', hparams_nlinear2),
      LinearFullPosteriorSampling('LinFullPost', hparams_linear),
      BootstrappedBNNSampling('BootRMS', hparams_rms),
      ParameterNoiseSampling('ParamNoise', hparams_pnoise),
      PosteriorBNNSampling('BBAlphaDiv', hparams_alpha_div, 'AlphaDiv'),
      #PosteriorBNNSampling('MultitaskGP', hparams_gp, 'GP'),
  ]

  # Run contextual bandit problem
  t_init = time.time()
  results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
  _, h_rewards, h_accumulative_rewards, h_regrets = results

  #time_step = np.arange(1, num_contexts+1)
  #plt.plot(time_step, h_accumulative_rewards[1:, 0])
  #plt.savefig('linear.png')
  '''
  with open('BBB_statlog_acc_rewards.txt', 'w') as f:
    for item in h_accumulative_rewards[1:, 1]:
      f.write("%s\n" % item)
  with open('Dropout_statlog_acc_rewards.txt', 'w') as f:
    for item in h_accumulative_rewards[1:, 0]:
      f.write("%s\n" % item)
  with open('BBB_statlog_regrets.txt', 'w') as f:
    for item in h_regrets[1:, 1]:
      f.write("%s\n" % item)
  with open('Dropout_statlog_regrets.txt', 'w') as f:
    for item in h_regrets[1:, 0]:
      f.write("%s\n" % item)
  '''
  # Display results
  display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, 'movie')

if __name__ == '__main__':
  app.run(main)
