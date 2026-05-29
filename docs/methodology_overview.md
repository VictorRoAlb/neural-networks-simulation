# Methodology Overview

## Problem framing

The original coursework studies multiclass prediction of burnout risk in a Gen Z population using tabular features and multilayer perceptrons.

The public version keeps the technical workflow that matters most:

- target normalization for the three classes `Low`, `Medium`, and `High`;
- leakage-safe feature preprocessing;
- stratified splitting into train, validation and test subsets;
- class-weighted optimization to address imbalance;
- model selection driven by validation Macro F1.

## Data split strategy

The experiment uses a two-step stratified split:

1. `90%` train+validation and `10%` final test.
2. validation carved out from the train+validation block so that the total validation share is also `10%`.

This keeps the test split untouched during model selection.

## Preprocessing

The preprocessing block is fit only on the training subset:

- numerical columns are standardized with `StandardScaler`;
- categorical and boolean columns are encoded with `OneHotEncoder(handle_unknown="ignore")`;
- the resulting matrices are reused for validation and test without refitting.

## Model families compared

The executed comparison includes:

- a linear softmax baseline;
- shallow one-hidden-layer networks trained with SGD;
- deeper five-layer MLP variants with `ReLU`, `tanh`, `sigmoid`, and `ELU`;
- deep variants with and without dropout;
- a Keras Tuner shallow-network search that is later retrained under the same common protocol.

## Training protocol

All candidate models share:

- the same maximum epoch budget;
- the same batch size;
- the same class weights;
- the same early-stopping logic;
- the same learning-rate scheduling logic;
- the same validation-driven stopping criterion.

Macro F1 is computed on both train and validation splits at the end of each epoch through a custom callback. That callback makes the training curves more informative for an imbalanced multiclass problem than accuracy alone.

## Selection rule

The best model is selected using `best_val_macro_f1`.

Test metrics are reported only after model selection is complete. This is one of the strongest aspects of the notebook because it keeps the validation and test roles clearly separated.

## Public artifacts

The repository stores:

- summary CSV tables with the model-comparison results;
- a compact experiment-configuration JSON file;
- report-ready figures used in the final coursework write-up;
- a clean Python script that mirrors the original notebook workflow.
