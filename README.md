# Sketched SGD

## Installation
Dependencies: `pytorch`, `numpy`, and `csvec` (https://github.com/nikitaivkin/csh). Tested with `torch==1.0.1` and `numpy==1.15.3`, but this should work with a range of versions.

`git clone` the repository to your local machine, move to the directory containing `setup.py`, then run
```
pip install -e .
```
to install this package.

## Description

This is the code accompanying the paper ``Communication-efficient distributed SGD with Sketching'' by Nikita Ivkin, Daniel Rothchild, Enayat Ullah, Vladimir Braverman, Ion Stoica, and Raman Arora (https://arxiv.org/abs/1903.04488)

This package contains three classes that can be used to carry out Sketched SGD in the simulated distributed setting. 
A simple example script is included showing how to train a model using this framework on FEMNIST.


