# OASIS Dataset Training Results Analysis

## Overview

This document provides an analysis of the recent model training run on the OASIS dataset, based on the logs generated on **March 3, 2026** (`train_oasis_trial1_20260303_030313.log`).

> Important: the March 3 run reported below was evaluating **edge reconstruction / link prediction quality inside dynamic brain graphs**, not dementia diagnosis classification. The original training pipeline did not use the OASIS class labels in the loss, so the AUCROC and AP values below should not be interpreted as Alzheimer's classification accuracy.

## Dataset and Configuration

- **Dataset:** OASIS (Parameters: `w-15`, `s-5`, `m-correlation`, `p-10`, `g-15`)
- **Total Subjects:** 1410 (each with 225 nodes)
- **Data Split:** 1128 Train / 282 Eval (Held-out)
- **Early Stopping:** Triggered if no improvement in validation Negative Log-Likelihood (NLL) for 15 evaluation steps (300 epochs).

## Training Progression

- The training resumed from an existing checkpoint at **Epoch 180** (Previous Best Valid NLL: 5.1889).
- At **Epoch 200**, the model hit its peak performance for validation data and corresponding test data.
- The training continued until **Epoch 500**, where **Early Stopping** was triggered, as validation NLL failed to improve for 15 consecutive evaluation checks (300 epochs).
- Between Epoch 200 and Epoch 500, the model exhibited steady minimization of training losses (KL divergences and NLL), but the validation and test NLLs gradually increased, clearly indicating **overfitting** to the training dataset.

## Key Edge-Reconstruction Metrics at Best Validation (Epoch 200)

The best validation checkpoint was saved at this epoch due to achieving the lowest Validation NLL.

| Metric                     | Train  | Validation | Test       |
| :------------------------- | :----- | :--------- | :--------- |
| **NLL**                    | 5.1650 | **5.1404** | **5.1360** |
| **Edge AUCROC**            | 0.6077 | 0.6114     | 0.6170     |
| **Average Precision (AP)** | 0.5697 | 0.5732     | 0.5729     |

## Final Edge-Reconstruction Metrics before Early Stopping (Epoch 500)

By the end of the training, the model degraded in generalizability.

| Metric                     | Train  | Validation | Test   |
| :------------------------- | :----- | :--------- | :----- |
| **NLL**                    | 5.4897 | 5.4699     | 5.4581 |
| **Edge AUCROC**            | 0.5868 | 0.5846     | 0.5895 |
| **Average Precision (AP)** | 0.5439 | 0.5392     | 0.5398 |

## Output Files Checkpoints

The following checkpoints have been saved and are ready for inference or future retraining:

- `/mnt/trainingresults/models_oasis_1/checkpoint_best_valid.pt` (Best generalization, from Epoch 200)
- `/mnt/trainingresults/models_oasis_1/checkpoint_best_train.pt`
- `/mnt/trainingresults/models_oasis_1/checkpoint_latest.pt` (From Epoch 500)

## Conclusion and Recommendations

1. **Metric Scope**: The model achieved an edge-reconstruction AUCROC of ~0.617 on held-out graph edges, not a dementia classification AUCROC. A separate label-aware objective or downstream classifier is required for Alzheimer's diagnosis performance claims.
2. **Overfitting**: The model aggressively overfits after Epoch 200. The divergence between train NLL and valid NLL becomes large. Consider increasing regularization techniques (e.g., dropout, weight decay) or tuning the capacity of the model to prevent such rapid overfitting in later epochs.
3. **Use the Right Checkpoint**: For any downstream evaluation, strictly use `checkpoint_best_valid.pt` rather than `checkpoint_latest.pt`.
