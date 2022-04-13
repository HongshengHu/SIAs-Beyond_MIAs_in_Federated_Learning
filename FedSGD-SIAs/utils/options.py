import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--num_samples', type=int, default=100,
                        help="number of samples from each local training set: N")
    parser.add_argument('--alpha', type=float, default=1, help="level of non-iid data distribution: alpha")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="testing batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    # other arguments
    parser.add_argument('-c', '--checkpoint', default='checkpoint/synthetic', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--manualseed', type=int, default=42, help='manual seed')
    parser.add_argument('--dataset', type=str, default='Synthetic', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--all_clients', default=True, action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args
