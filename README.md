# Emotion-Guided Cross-Attention for Multi-Task Abusive Language Detection

A PyTorch implementation of a multi-task transformer model for abusive language detection that jointly learns **fine-grained emotional abuse classification** and **binary relationship abuse detection**. The model leverages **emotion-guided cross-attention** to improve abuse detection by incorporating emotional context into the classification process.

## Overview

Online abusive language is often subtle and context-dependent, making it difficult for traditional binary classifiers to detect. This project proposes a **multi-task learning (MTL)** framework that simultaneously predicts:

- Fine-grained emotional abuse behaviors (multi-label classification)
- Binary relationship abuse (binary classification)

The emotional representations learned from the first task guide the second task through a **cross-attention mechanism**, enabling the model to better recognize nuanced abusive language.

## Features

- Multi-task learning architecture
- Shared Transformer encoder
- Emotion-guided cross-attention
- Multi-label emotion classification
- Binary abuse detection
- Focal Loss for handling class imbalance
- Uncertainty-weighted joint loss optimization
- Dynamic threshold tuning for multi-label prediction
- Ablation study support

## Model Architecture

```
                    Input Text
                         │
              Shared Transformer Encoder
                         │
                 Mean Pooling Layer
                         │
            ┌────────────┴────────────┐
            │                         │
   Emotion Classification      Cross-Attention
      (Task 1)                     │
            │                      │
            └────────► Guided Representation
                                   │
                          Abuse Classification
                               (Task 2)
```

## Datasets

The model is evaluated using:

- **Reddit Relationship Abuse Dataset**
  - Binary relationship abuse detection

- **Unhealthy Comments Corpus (UCC)**
  - Multi-label emotional abuse classification

## Training

The model incorporates several techniques to improve performance:

- Focal Loss for imbalanced multi-label learning
- Binary Cross-Entropy for abuse classification
- Uncertainty-based loss weighting
- Gradient accumulation
- Dynamic threshold optimization on the validation set

## Results

### Binary Abuse Detection

| Model | F1 Score | Accuracy |
|--------|---------:|---------:|
| Full Model | **0.9052** | **0.9087** |
| Without Emotion Guidance | 0.8727 | 0.8838 |

The ablation study demonstrates that emotion-guided cross-attention significantly improves abuse detection performance.


## Future Work

- Improve implicit abuse detection
- Incorporate larger and more diverse datasets
- Explore advanced attention mechanisms
- Extend to multilingual abusive language detection
