# SIAs-Beyond_MIAs_in_Federated_Learning
This repository contains the code for the experiments of the paper "Source Inference Attacks: Beyond Membership Inference Attacks in Federated Learning"

# Requirement
* torch==1.8.1
* numpy==1.18.1
* torchvision==0.9.1
* sklearn==0.22.1

# Dataset
We evaluate SIAs on three typical federated learning frameworks, i.e., [FedSGD](https://arxiv.org/abs/1602.05629), [FedAvg](https://arxiv.org/abs/1602.05629), and [FedMD](https://arxiv.org/abs/1910.03581).

We use Synthetic, MNIST, FEMNIST, CIFAR-10, CIFAR-100, and Purchase datasets for the experiments. In particular, we use Synthetic, MNIST, CIFAR-10, and Purchase for evaluating SIAs in FedSGD and FedAvg. We use MNIST, FEMNIST, CIFAR-10, and CIFAR-100 for evaluating SIAs in FedMD.

* For MNIST, CIFAR-10, and CIFAR-100 datasets, you can directly run the code and the dataset will be downloaded automatically.
* For Purchase-100, please first dowanlad it from [here](https://drive.google.com/drive/folders/1FBJ6c8v9pM9kO1tX19ccmd3noZRC2JBh?usp=sharing), and then put it in the "data" subfolder of the "FedSGD-SIAs" and the "FedAvg-SIAs" folder
* For Synthetic, FEMNIST datasets, we have uploaded them to the repository.

# Experiment on FedSGD
You can try different `--alpha` (non-IID data distribution) to evaluate how non-IID distribution affects the attack performance. For `Synthetic` and `Purchase` datasets, we set `--model=mlp`. For `MNIST` and `CIFAR-10` dataset, we set `--model=cnn`.
```python
python main_fed.py --dataset=Synthetic --model=mlp --alpha=1 
```
