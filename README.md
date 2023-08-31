# Classification of Breast Cancer Histopathology Images using a Modified Supervised Contrastive Learning Method

Implemented a self supervised method
based on multiple magnifications of images and apply histopathology augmentations, which
reached state-of-the-art results in detecting malignant tumors on BreahHis dataset.
# Requirement
This repository code is compaitible with Python 3.6 and 3.8, Pytorch 1.2.0, and Torchvision 0.4.0.

# Commands

**Self-supervised pretraining (Assuming in directory 'src')** 

```python -m self_supervised.experiments.pretrain_MPCS --data_fold <'train_data_fold_path'> --pair_sampling <'OP'/'RP'/'FP'> --LR <learning_rate - 0.00001> --epoch <150> --description <'experiment_name'>```
**OP - Ordered Pair, RP - Random Pair, and FP - Fixed Pair

**Fintuning using MPCS pretrained Efficient-net b2 on BreakHis (Assuming in directory 'src')**

```python -m supervised.experiments.finetune_mpcs_on_breakhis2 --train_data_fold <'train_data_fold_path'> --test_data_fold <'test_data_fold_path'> --magnification <'40x'/'100x'/'200x'/'400x'>  --model_path <'pretrained model path'> --LR <learning_rate - 0.00002> --epoch <150> --description <'experiment_name'>```

**Evaluation**

```python - m supervised.evaluation.evaluation <check paramters in file itself>```

