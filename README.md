# Self-Pruning Neural Network

## Overview
This project implements a neural network that learns to prune itself during training using learnable gates and L1 regularization.

## Features
- Custom PrunableLinear layer
- Automatic pruning during training
- Sparsity vs accuracy trade-off

## How to Run
Open the notebook:
pruning.ipynb

## Results
output.png

## Key Insight
L1 regularization pushes gate values to zero, pruning unnecessary weights.
