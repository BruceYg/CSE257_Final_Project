# GLCB-report

This repository contains implementation for <em> Report on Online Learning in Contextual Bandits using Gated Linear Networks (NeurIPS 2020) </em>, which is a course project for CSE257 at UCSD.

The original paper is [Online Learning in Contextual Bandits using Gated Linear Networks](https://arxiv.org/abs/2002.11611).

Part of the implementation follows <https://github.com/deepmind/deepmind-research/tree/master/gated_linear_networks> and <https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits>.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages.

```bash
pip3 install --upgrade pip setuptools wheel
pip3 install --upgrade tf_slim
pip3 install jax
pip3 install chex
pip3 install dm-haiku
pip3 install rlax
```

## Usage

Run the following files for different experiments.

```bash
# Run GLCB on datasets in Section 2
python3 GLCB.py

# Run Thompson Sampling based algorithms
# on datasets in Section 2
python3 baseline.py

# Run GLCB on MovieLens 100K
python3 GLCB_movie.py

# Run Thompson Sampling based algorithms
# on MovieLens 100K
python3 baseline_movie.py
```

Make sure to download the datasets using the links below and put the data files inside ./contextual_bandits/datasets.

[Adult](https://storage.googleapis.com/bandits_datasets/adult.full)\
[Census](https://storage.googleapis.com/bandits_datasets/USCensus1990.data.txt)\
[Covertype](https://storage.googleapis.com/bandits_datasets/covtype.data)\
[Statlog](https://storage.googleapis.com/bandits_datasets/shuttle.trn)\
[MovieLens](https://files.grouplens.org/datasets/movielens/ml-100k.zip) Unzip and put the folder in ./
