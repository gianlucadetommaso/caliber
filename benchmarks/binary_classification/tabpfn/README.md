# Results

These results show the performance and confidence calibration of [XGBoost](https://xgboost.readthedocs.io/en/stable/) and [TabPFN](https://github.com/PriorLabs/TabPFN) on the Breast Cancer binary classification dataset, as well as the performance and confidence calibration of post-hoc calibration algorithms built on top of TabPFN.

Training and test datasets follow a 90%/10% split. To make the evaluation as fair as possible, results for both XGBoost and TabPFN (first two rows) are for models trained on the full 90% of data. For all the following calibration models, instead, we split the training data 50%/50% into sub-training and calibration dataset. A TabPFN is trained on the sub-training datasets, and its outputs are used to train each calibration model on the calibration dataset.

```
╭──────────────────────────┬────────────┬─────────────────────┬─────────────┬──────────┬───────────────┬─────────────────┬───────────────┬────────────┬────────────╮
│ BREAST CANCER            │   accuracy │   balanced_accuracy │   precision │   recall │   positive_F1 │   cross-entropy │   Brier score │       ASCE │        ECE │
├──────────────────────────┼────────────┼─────────────────────┼─────────────┼──────────┼───────────────┼─────────────────┼───────────────┼────────────┼────────────┤
│ XGBClassifier            │   0.964912 │            0.971429 │           1 │ 0.942857 │      0.970588 │      0.0587525  │    0.0195279  │ 0.0105882  │ 0.0302765  │
│ TabPFNClassifier         │   0.982456 │            0.985714 │           1 │ 0.971429 │      0.985507 │      0.0279245  │    0.00645344 │ 0.00620463 │ 0.0232546  │
│ beta                     │   1        │            1        │           1 │ 1        │      1        │      0.0313291  │    0.00612025 │ 0.00591774 │ 0.0274637  │
│ histogram_binning        │   1        │            1        │           1 │ 1        │      1        │      0.028611   │    0.00408211 │ 0.00371846 │ 0.026081   │
│ isotonic_regression      │   1        │            1        │           1 │ 1        │      1        │      0.00711354 │    0.00194932 │ 0.00194932 │ 0.00584795 │
│ iterative_linear_binning │   1        │            1        │           1 │ 1        │      1        │      0.0155138  │    0.00393781 │ 0.00389717 │ 0.012729   │
╰──────────────────────────┴────────────┴─────────────────────┴─────────────┴──────────┴───────────────┴─────────────────┴───────────────┴────────────┴────────────╯
```

On this dataset, results demonstrate TabPFN is superior to XGBoost on both performance and calibration metrics. However, all calibration models improve performance metrics, and several improve the calibration metrics. The combination of TabPFN and Isotonic Regression performs the best.
