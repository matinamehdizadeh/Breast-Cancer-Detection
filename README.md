# Classification of Breast Cancer Histopathology Images using a Modified Supervised Contrastive Learning Method

Implemented a modified supervised contrastive learning method
based on the labels and similarity between images. Also, apply histopathology augmentations, which
reached state-of-the-art results in detecting malignant tumors on BreahHis dataset.
# Requirement
This repository code is compaitible with Python 3.6 and 3.8, Pytorch 1.2.0, and Torchvision 0.4.0.

# Commands

**Modified-Supervised-Contrastive Representation Learning Phase1 (Assuming in directory 'src')** 

```python -m self_supervised_phase1/run.py --data_fold <'train_data_fold_path'>  --LR <learning_rate - 0.00001> --epoch <150> --description <'experiment_name'>```

**Modified-Supervised-Contrastive Representation Learning Phase2 (Assuming in directory 'src')** 

```python -m self_supervised_phase2/run.py --data_fold <'train_data_fold_path'>  --LR <learning_rate - 0.00001> --epoch <150> --description <'experiment_name'>```
**Calculate Similarity Phase (Assuming in directory 'src')** 

```python -m similarity/run.py --data_fold <'train_data_fold_path'>  --LR <learning_rate - 0.00001> --epoch <150> --description <'experiment_name'>```

**Fintuning using pretrained Efficient-net b2 on BreakHis (Assuming in directory 'src')**

```python -m supervised.experiments.breakhis --train_data_fold <'train_data_fold_path'> --test_data_fold <'test_data_fold_path'> --magnification <'40x'/'100x'/'200x'/'400x'>  --model_path <'pretrained model path'> --LR <learning_rate - 0.00002> --epoch <150> --description <'experiment_name'>```

**Fintuning using pretrained Efficient-net b2 on BACH (Assuming in directory 'src')**

```python -m supervised.experiments.bach --train_data_fold <'train_data_fold_path'> --test_data_fold <'test_data_fold_path'> --magnification <'40x'/'100x'/'200x'/'400x'>  --model_path <'pretrained model path'> --LR <learning_rate - 0.00002> --epoch <150> --description <'experiment_name'>```

**Evaluation**

```python - m supervised.evaluation.evaluation ```

