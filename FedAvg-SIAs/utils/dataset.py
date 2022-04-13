import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from .sampling import sample_dirichlet_train_data


def get_dataset(args):
    if args.dataset == 'CIFAR10':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(train_dataset, args.num_users, args.num_samples,
                                                                        args.alpha)

    elif args.dataset == 'MNIST':
        data_dir = './data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        # sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(train_dataset, args.num_users, args.num_samples,
                                                                        args.alpha)


    elif args.dataset == 'Purchase':

        purchase_x_path = './data/purchase/purchase_x.npy'
        purchase_y_path = './data/purchase/purchase_y.npy'

        X = np.load(purchase_x_path)
        Y = np.load(purchase_y_path)

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_dataset = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()))
        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()))
        # sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(train_dataset, args.num_users, args.num_samples,
                                                                        args.alpha)

    elif args.dataset == 'Synthetic':
        data_dir = './data/synthetic/synthetic.npz'
        synt_0 = np.load(data_dir)
        X = synt_0['x'].astype(np.float64)
        Y = synt_0['y'].astype(np.int32)

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_dataset = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()))
        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()))
        # sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(train_dataset, args.num_users, args.num_samples,
                                                                        args.alpha)

    else:
        train_dataset = []
        test_dataset = []
        dict_party_user, dict_sample_user = {}, {}
        print('+' * 10 + 'Error: unrecognized dataset' + '+' * 10)
    return train_dataset, test_dataset, dict_party_user, dict_sample_user


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : sgd')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.dataset == 'syn0' or args.dataset == 'syn1':
        print(f'   Synthetic dataset for IID setting: {args.dataset}' + f' has {args.num_classes} classes')
    else:
        print(f'   Realistic dataset for Non-IID setting:{args.dataset}'f' has {args.num_classes} classes')
        print(f'   Level of non-iid data distribution:{args.alpha}')
    print(f'    Number of users  : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
