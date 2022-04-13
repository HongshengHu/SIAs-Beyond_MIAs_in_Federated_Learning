import copy


def FedAvg(w, weight):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += weight[i] * w[i][k]

    return w_avg
