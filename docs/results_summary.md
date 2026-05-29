# Results Summary

## Final ranking

The strongest configuration in the published comparison is:

- `deep_sigmoid_adam_dropout025`
- hidden layers: `512-256-128-64-32`
- activation: `sigmoid`
- optimizer: `Adam`
- dropout: `0.25`

It was selected because it produced the highest validation Macro F1 among the compared models.

## Key metrics

Final published values for the selected model:

- best validation Macro F1: `0.9734`
- test accuracy: `0.9980`
- test Macro F1: `0.9986`
- `Low` class F1: `1.0000`
- `Medium` class F1: `0.9978`
- `High` class F1: `0.9981`

## Interpretation

Several patterns stand out in the public results:

- deep architectures clearly outperform the linear baseline and the weaker SGD configurations;
- a shallow model found with Keras Tuner is competitive, but still trails the best deep model in Macro F1;
- the class-weighting strategy appears especially important because the `Low` class has very limited support;
- the final confusion matrix shows only a minimal amount of residual confusion, concentrated in the `High` versus `Medium` boundary.

## Why these figures are highlighted

The repository emphasizes four report figures because together they tell the most complete public story:

1. class distribution and class weights explain the imbalance challenge;
2. learning curves show how the selected model converges;
3. the model-comparison plot shows why the selected network was preferred;
4. the confusion matrix closes the loop with class-wise final performance.
