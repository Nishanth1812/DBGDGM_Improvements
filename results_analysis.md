# OASIS Training Results

These runs are mostly measuring edge reconstruction, not Alzheimer's diagnosis. The training objective does not use the OASIS labels directly, so edge AUCROC and AP should be read as graph prediction metrics. Diagnosis numbers below come from a separate downstream evaluation on the learned embeddings.

## Run 1

This run comes from the March 3 log `train_oasis_trial1_20260303_030313.log`. It used 1410 subjects, resumed from epoch 180, found its best validation checkpoint at epoch 200, and then kept training until early stopping at epoch 500. The pattern is straightforward: the model kept fitting the training data, while validation quality drifted the wrong way.

Best validation checkpoint at epoch 200:

| Metric                     | Train  | Validation | Test   |
| :------------------------- | :----- | :--------- | :----- |
| **NLL**                    | 5.1650 | 5.1404     | 5.1360 |
| **Edge AUCROC**            | 0.6077 | 0.6114     | 0.6170 |
| **Average Precision (AP)** | 0.5697 | 0.5732     | 0.5729 |

Final checkpoint before early stopping at epoch 500:

| Metric                     | Train  | Validation | Test   |
| :------------------------- | :----- | :--------- | :----- |
| **NLL**                    | 5.4897 | 5.4699     | 5.4581 |
| **Edge AUCROC**            | 0.5868 | 0.5846     | 0.5895 |
| **Average Precision (AP)** | 0.5439 | 0.5392     | 0.5398 |

Takeaway: Run 1 clearly overfit after epoch 200. If this run is used at all, `checkpoint_best_valid.pt` is the only checkpoint worth keeping for evaluation.

## Run 2

This run comes from the March 8 log `train_oasis_trial1_20260308_061215.log`. It used the subject-level cache with 365 subjects, batch size 32, learning rate `1e-4`, weight decay `1e-4`, and `classification_weight=0.0`. It started from scratch, reached its best validation checkpoint at epoch 85, and stopped early at epoch 110.

Best validation checkpoint at epoch 85:

| Metric                     | Train  | Validation | Test   |
| :------------------------- | :----- | :--------- | :----- |
| **NLL**                    | 5.3004 | 5.3017     | 5.2979 |
| **Edge AUCROC**            | 0.6060 | 0.6083     | 0.6097 |
| **Average Precision (AP)** | 0.5700 | 0.5703     | 0.5736 |

Final checkpoint before early stopping at epoch 110:

| Metric                     | Train  | Validation | Test   |
| :------------------------- | :----- | :--------- | :----- |
| **NLL**                    | 5.2999 | 5.3018     | 5.2973 |
| **Edge AUCROC**            | 0.6063 | 0.6082     | 0.6104 |
| **Average Precision (AP)** | 0.5702 | 0.5700     | 0.5746 |

Downstream diagnosis metrics from the saved embeddings:

| Metric                | Best Validation | Final |
| :-------------------- | :-------------- | :---- |
| **Accuracy**          | 0.4904 +/- 0.0271 | 0.4915 +/- 0.0456 |
| **Balanced Accuracy** | 0.2682 +/- 0.0445 | 0.2822 +/- 0.0343 |
| **Macro F1**          | 0.2437 +/- 0.0240 | 0.2503 +/- 0.0263 |
| **Macro AUC OVR**     | 0.5655 +/- 0.0462 | 0.5792 +/- 0.0297 |

Takeaway: Run 2 is much more stable than Run 1, but the diagnosis signal is still weak. Balanced accuracy stays below 0.30, and the rarest class has only 2 subjects, so these diagnosis metrics should be treated cautiously. For comparisons or downstream use, `checkpoint_best_valid.pt` is still the right checkpoint.

## Bottom Line

Run 2 is the cleaner result. It avoids the obvious late overfitting seen in Run 1 and keeps edge-reconstruction quality roughly flat through training. That said, neither run supports a strong diagnosis claim yet. If the goal is Alzheimer's classification, the next improvement needs to come from the modeling setup or evaluation design, not from squeezing a few more epochs out of the current objective.
