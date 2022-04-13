import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
import scipy.io as sio
import torchvision
import random


def normalization(x_img_train, x_img_test):
    mean = np.mean(x_img_train, axis=(0, 1, 2))
    std = np.std(x_img_train, axis=(0, 1, 2))

    x_img_train = (x_img_train - mean) / (std + 1e-7)
    x_img_test = (x_img_test - mean) / (std + 1e-7)

    return x_img_train, x_img_test


def load_MNIST_data(standarized=False, verbose=False):
    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True,
                                          download=True)
    devset = torchvision.datasets.MNIST(root='./data/mnist', train=False,
                                        download=True)

    X_train = trainset.data.numpy()
    X_test = devset.data.numpy()

    y_train = np.array(trainset.targets)
    y_test = np.array(devset.targets)

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]

    X_train, X_test = normalization(X_train, X_test)

    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    return X_train, y_train, X_test, y_test


def load_EMNIST_data(file, verbose=False, standarized=False):
    """
    file should be the downloaded EMNIST file in .mat format.
    """
    mat = sio.loadmat(file)
    data = mat["dataset"]

    writer_ids_train = data['train'][0, 0]['writers'][0, 0]
    writer_ids_train = np.squeeze(writer_ids_train)
    X_train = data['train'][0, 0]['images'][0, 0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order="F")
    y_train = data['train'][0, 0]['labels'][0, 0]
    y_train = np.squeeze(y_train)
    y_train -= 1  # y_train is zero-based

    writer_ids_test = data['test'][0, 0]['writers'][0, 0]
    writer_ids_test = np.squeeze(writer_ids_test)
    X_test = data['test'][0, 0]['images'][0, 0]
    X_test = X_test.reshape((X_test.shape[0], 28, 28), order="F")
    y_test = data['test'][0, 0]['labels'][0, 0]
    y_test = np.squeeze(y_test)
    y_test -= 1  # y_test is zero-based

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255

    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]

    X_train, X_test = normalization(X_train, X_test)

    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    if verbose == True:
        print("EMNIST-letter dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test, writer_ids_train, writer_ids_test


def load_CIFAR_data(data_type="CIFAR10", label_mode="fine",
                    standarized=True, verbose=False):
    if data_type == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True)
        devset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                              download=True)

        X_train = trainset.data
        X_test = devset.data

        y_train = np.array(trainset.targets)
        y_test = np.array(devset.targets)

    elif data_type == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True,
                                                 download=True)
        devset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False,
                                               download=True)
        X_train = trainset.data
        X_test = devset.data

        y_train = np.array(trainset.targets)
        y_test = np.array(devset.targets)

    else:
        print("Unknown Data type. Stopped!")
        return None

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255

    X_train, X_test = normalization(X_train, X_test)

    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    return X_train, y_train, X_test, y_test


def build_classes_dict(dataset):
    classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if torch.is_tensor(label):
            label = label.numpy()[0]
        else:
            label = label
        if label in classes:
            classes[label].append(ind)
        else:
            classes[label] = [ind]

    return classes


def sample_dirichlet_train_data(dataset, no_participants, no_samples, manualseed, alpha=0.1):
    random.seed(manualseed)
    data_classes = build_classes_dict(dataset)
    class_size = len(data_classes[0])
    per_participant_list = defaultdict(list)
    per_samples_list = defaultdict(list)
    no_classes = len(data_classes.keys())  # for cifar: 10

    image_nums = []
    for n in range(no_classes):
        image_num = []
        random.shuffle(data_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))

        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))

            sampled_list = data_classes[n][:min(len(data_classes[n]), no_imgs)]
            image_num.append(len(sampled_list))
            per_participant_list[user].extend(sampled_list)
            data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs):]
        image_nums.append(image_num)

    for i in range(len(per_participant_list)):
        no_samples = min(no_samples, len(per_participant_list[i]))

    for i in range(len(per_participant_list)):
        sample_index = np.random.choice(len(per_participant_list[i]), no_samples,
                                        replace=False)
        per_samples_list[i].extend(np.array(per_participant_list[i])[sample_index])

    return per_participant_list, per_samples_list


def generate_partial_data(X, y, class_in_use=None, verbose=False):
    if class_in_use is None:
        idx = np.ones_like(y, dtype=bool)
    else:

        idx = [y == i for i in class_in_use]

        idx = np.any(idx, axis=0)

    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose == True:
        print("X shape :", X_incomplete.shape)
        print("y shape :", y_incomplete.shape)
    return X_incomplete, y_incomplete


def generate_dirc_private_data(X, y, alpha, N_parties=10, classes_in_use=range(11),
                               num_source_samples=100, manualseed=42, size_limit=False):
    class_size = len(y[y == y[0]])

    priv_data = [None] * N_parties
    source_data = [None] * N_parties
    combined_idx = np.array([], dtype=np.int16)
    source_idx = np.array([], dtype=np.int16)
    for cls in classes_in_use:
        random.seed(manualseed)
        idx = np.where(y == cls)[0]

        if size_limit == True:
            idx = idx[:500]
            class_size = 500
        random.shuffle(idx)

        sampled_probabilities = class_size * np.random.dirichlet(np.array(N_parties * [alpha]))

        combined_idx = np.r_[combined_idx, idx]
        for user_idx in range(N_parties):
            num_imgs = int(round(sampled_probabilities[user_idx]))

            sampled_list = idx[:min(class_size, num_imgs)]

            if priv_data[user_idx] is None:
                tmp = {}
                tmp["X"] = X[sampled_list]
                tmp["y"] = y[sampled_list]
                tmp["idx"] = sampled_list
                priv_data[user_idx] = tmp
            else:
                priv_data[user_idx]['idx'] = np.r_[priv_data[user_idx]["idx"], sampled_list]
                priv_data[user_idx]["X"] = np.vstack([priv_data[user_idx]["X"], X[sampled_list]])
                priv_data[user_idx]["y"] = np.r_[priv_data[user_idx]["y"], y[sampled_list]]

            idx = idx[min(class_size, num_imgs):]

    for i in range(N_parties):
        num_source_samples = min(num_source_samples, len(priv_data[i]["idx"]))

    for i in range(N_parties):
        sample_index = np.random.choice(len(priv_data[i]["idx"]), num_source_samples,
                                        replace=False)

        tmp = {}
        tmp["X"] = priv_data[i]["X"][sample_index]
        tmp["y"] = priv_data[i]["y"][sample_index]
        tmp["idx"] = priv_data[i]["idx"][sample_index]
        source_data[i] = tmp
        source_idx = np.r_[source_idx, tmp["idx"]]

    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]

    total_source_data = {}
    total_source_data["idx"] = source_idx
    total_source_data["X"] = X[source_idx]
    total_source_data["y"] = y[source_idx]
    return priv_data, total_priv_data, source_data, total_source_data


def generate_bal_private_data(X, y, N_parties=10, classes_in_use=range(11),
                              N_samples_per_class=20, data_overlap=False):
    priv_data = [None] * N_parties
    combined_idx = np.array([], dtype=np.int16)
    for cls in classes_in_use:
        np.random.seed(121)
        idx = np.where(y == cls)[0]

        idx = np.random.choice(idx, N_samples_per_class * N_parties,
                               replace=data_overlap)
        combined_idx = np.r_[combined_idx, idx]
        for i in range(N_parties):
            idx_tmp = idx[i * N_samples_per_class: (i + 1) * N_samples_per_class]
            if priv_data[i] is None:
                tmp = {}
                tmp["X"] = X[idx_tmp]
                tmp["y"] = y[idx_tmp]
                tmp["idx"] = idx_tmp
                priv_data[i] = tmp
            else:
                priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]

    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return priv_data, total_priv_data


def generate_alignment_data(X, y, N_alignment=3000):
    split = StratifiedShuffleSplit(n_splits=1, train_size=N_alignment, random_state=42)
    if N_alignment == "all":
        alignment_data = {}
        alignment_data["idx"] = np.arange(y.shape[0])
        alignment_data["X"] = X
        alignment_data["y"] = y
        return alignment_data
    for train_index, _ in split.split(X, y):
        X_alignment = X[train_index]
        y_alignment = y[train_index]
    alignment_data = {}
    alignment_data["idx"] = train_index
    alignment_data["X"] = X_alignment
    alignment_data["y"] = y_alignment

    return alignment_data


def generate_EMNIST_writer_based_data(X, y, writer_info, N_priv_data_min=30,
                                      N_parties=5, classes_in_use=range(6)):
    mask = None
    mask = [y == i for i in classes_in_use]
    mask = np.any(mask, axis=0)

    df_tmp = None
    df_tmp = pd.DataFrame({"writer_ids": writer_info, "is_in_use": mask})
    groupped = df_tmp[df_tmp["is_in_use"]].groupby("writer_ids")

    # organize the input the data (X,y) by writer_ids.
    # That is, 
    # data_by_writer is a dictionary where the keys are writer_ids,
    # and the contents are the correcponding data. 
    # Notice that only data with labels in class_in_use are included.
    data_by_writer = {}
    writer_ids = []
    for wt_id, idx in groupped.groups.items():
        if len(idx) >= N_priv_data_min:
            writer_ids.append(wt_id)
            data_by_writer[wt_id] = {"X": X[idx], "y": y[idx],
                                     "idx": idx, "writer_id": wt_id}

    # each participant in the collaborative group is assigned data 
    # from a single writer.
    ids_to_use = np.random.choice(writer_ids, size=N_parties, replace=False)
    combined_idx = np.array([], dtype=np.int64)
    private_data = []
    for i in range(N_parties):
        id_tmp = ids_to_use[i]
        private_data.append(data_by_writer[id_tmp])
        combined_idx = np.r_[combined_idx, data_by_writer[id_tmp]["idx"]]
        del id_tmp

    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return private_data, total_priv_data


def generate_imbal_CIFAR_private_data(X, y, y_super, classes_per_party, N_parties,
                                      samples_per_class=7):
    priv_data = [None] * N_parties
    combined_idxs = []
    count = 0
    for subcls_list in classes_per_party:
        idxs_per_party = []
        for c in subcls_list:
            idxs = np.flatnonzero(y == c)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            idxs_per_party.append(idxs)
        idxs_per_party = np.hstack(idxs_per_party)
        combined_idxs.append(idxs_per_party)

        dict_to_add = {}
        dict_to_add["idx"] = idxs_per_party
        dict_to_add["X"] = X[idxs_per_party]
        dict_to_add["y"] = y_super[idxs_per_party]
        priv_data[count] = dict_to_add
        count += 1

    combined_idxs = np.hstack(combined_idxs)
    total_priv_data = {}
    total_priv_data["idx"] = combined_idxs
    total_priv_data["X"] = X[combined_idxs]

    total_priv_data["y"] = y_super[combined_idxs]
    return priv_data, total_priv_data
