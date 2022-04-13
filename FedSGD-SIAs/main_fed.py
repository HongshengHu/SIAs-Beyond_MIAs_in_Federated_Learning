import copy
import numpy as np
import torch
import os
from models.Fed import FedAvg
from models.Sia import SIA
from models.Nets import MLP, Mnistcnn, CifarCnn
from models.Update import LocalUpdate
from models.test import test_img
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from utils.logger import Logger, mkdir_p

if __name__ == '__main__':
    args = args_parser()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # record the experimental results
    logger = Logger(os.path.join(args.checkpoint, 'log_seed{}.txt'.format(args.manualseed)))
    logger.set_names(['alpha', 'comm. round', 'ASR'])

    # parse args
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dataset_train, dataset_test, dict_party_user, dict_sample_user = get_dataset(args)

    # build model
    if args.model == 'cnn' and args.dataset == 'CIFAR10':
        net_glob = CifarCnn(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'MNIST':
        net_glob = Mnistcnn(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        dataset_train = dataset_train.dataset
        dataset_test = dataset_test.dataset
        data_size = dataset_train[0][0].shape

        for x in data_size:
            len_in *= x

        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    empty_net = net_glob

    print(net_glob)
    net_glob.train()

    size_per_client = []
    for i in range(args.num_users):
        size = len(dict_party_user[i])
        size_per_client.append(size)

    total_size = sum(size_per_client)
    size_weight = np.array(np.array(size_per_client) / total_size)

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    acc_loss_attack = []

    best_att_acc = 0
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx])

            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # implement source inference attack
        S_attack = SIA(args=args, w_locals=w_locals, dataset=dataset_train, dict_mia_users=dict_sample_user)
        attack_acc_loss = S_attack.attack(net=empty_net.to(args.device))

        logger.append([args.alpha, iter, attack_acc_loss])

        # save model for the epoch that achieve the max source inference accuracy
        if attack_acc_loss > best_att_acc:
            torch.save(w_locals, os.path.join(args.checkpoint, 'model_weight'))
            torch.save(dict_party_user, os.path.join(args.checkpoint, 'local_data'))

        best_att_acc = max(best_att_acc, attack_acc_loss)

        # update global weights
        w_glob = FedAvg(w_locals, size_weight)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        acc_train, loss_train_ = test_img(net_glob, dataset_train, args)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    logger.close()

    # testing
    net_glob.eval()
    exp_details(args)
    acc_train, loss_train_ = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print('Best attack accuracy: {:.2f}'.format(best_att_acc))
