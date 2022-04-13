import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def _safe_prob(probs, small_value=1e-30):
    return np.maximum(probs, small_value)

def uncertainty(probability, n_classes):
    uncert = []
    for i in range(len(probability)):
        unc = (-1 / np.log(n_classes)) * np.sum(probability[i] * np.log(_safe_prob(probability[i])))
        uncert.append(unc)
    return uncert


def entropy_modified(probability, target):
    entr_modi = []
    for i in range(len(probability)):
        ent_mod_1 = (-1) * (1 - probability[i][int(target[i])]) * np.log(_safe_prob(probability[i][int(target[i])]))
        probability_rest = np.delete(probability[i], int(target[i]))
        ent_mod_2 = -np.sum(probability_rest* np.log(_safe_prob(1 - probability_rest)))
        ent_mod = ent_mod_1 + ent_mod_2
        entr_modi.append(ent_mod)
    return entr_modi


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class SIA(object):
    def __init__(self, w_locals=None, dataset=None, cuda=True):
        self.w_locals = w_locals
        self.dataset = dataset
        self.gpu=cuda

    def attack(self):
        correct_loss = 0
        len_set = 0
        for idx in range(10):
            dataset_local=DataLoader(TensorDataset(torch.from_numpy(self.dataset[idx]["X"]).float(),
                                              torch.from_numpy(self.dataset[idx]["y"]).long()),batch_size=16,shuffle=False)


            y_loss_all = []

            # evaluate each party's training data on each party's model
            for local in range(10):

                y_losse = []

                idx_tensor = torch.tensor(idx)
                # net.load_state_dict(self.w_locals[local])
                net=self.w_locals[local]
                net.eval()
                for id, (data, target) in enumerate(dataset_local):
                    if self.gpu == True:
                        data, target = data.cuda(), target.cuda()
                        idx_tensor = idx_tensor.cuda()
                    log_prob = net(data)
                    # get the prediction loss of the sample
                    loss = nn.CrossEntropyLoss(reduction='none')
                    y_loss = loss(log_prob, target)
                    y_losse.append(y_loss.cpu().detach().numpy())

                y_losse = np.concatenate(y_losse).reshape(-1)
                y_loss_all.append(y_losse)

            device = torch.device("cuda:0")

            y_loss_all = torch.tensor(y_loss_all).to(device)

            # test if the owner party has the largest prediction probability
            # get the parties' index of the largest probability of each sample
            index_of_party_loss = y_loss_all.min(0, keepdim=True)[1]
            correct_local_loss = index_of_party_loss.eq(
                idx_tensor.repeat_interleave(len(dataset_local.dataset))).long().cpu().sum()

            correct_loss += correct_local_loss
            len_set += len(dataset_local.dataset)

        # calculate membership inference attack accuracy
        accuracy_loss = 100.00 * correct_loss / len_set

        print('\nTotal attack accuracy of prediction loss based attack: {}/{} ({:.2f}%)\n'.format(correct_loss, len_set,
                                                                                                  accuracy_loss))

        return accuracy_loss