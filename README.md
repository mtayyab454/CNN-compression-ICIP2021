# Compressing Deep CNNs Using Basis Representation and Spectral Fine-Tuning ([Link](https://arxiv.org/abs/2105.10436)).

Pytorch implementation of "Compressing Deep CNNs Using Basis Representation and Spectral Fine-Tuning" (ICIP 2021).

<div align=center><img src="img/framework.png" height = "60%" width = "70%"/></div>


## Citation
Please consider citing:

```
@inproceedings{tayyab2021SFT,
author={Tayyab, Muhammad and Khan, Fahad Ahmad and Mahalanobis, Abhijit},
booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
title={Compressing Deep CNNs Using Basis Representation and Spectral Fine-Tuning},
year={2021},
pages={3537-3541},
doi={10.1109/ICIP42928.2021.9506128}}
```

### Model Compression

Following commands can be used to reproduce the results presented in the paper. 

##### 1. Resnet56

| Flops         | Parameters      | Accuracy |
|---------------|-----------------|----------|
|89.80M(64.22%) | 0.32M(62.97%)   | 92.71%   | 

```shell
python run_cifar.py \
--jobid resnet56_test \
--arch resnet56 \
--dataset cifar10 \
--compress_rate :[6,4,4,6,4,4,4,4,4,4,4,4,4,13,4,10,6,4,4,12,18,16,4,15,4,16,4,12,7,13,4,15,4,18,4,12,4,32,26,36,16,32,13,29,23,32,16,36,10,23,13,20,10,13,7] \
--l2_weight 0.001 \
--add_bn True \
--epochs 120 \
--schedule 30 60 90 \
--lr 0.01
```
