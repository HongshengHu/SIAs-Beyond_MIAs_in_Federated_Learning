import os
import argparse
import sys
import torch
import numpy as np
from torch.utils.data import TensorDataset
from utils.data_utils import load_MNIST_data, load_EMNIST_data, generate_partial_data, generate_dirc_private_data
from FedMD_SIAs import FedMD_SIAs
from utils.Neural_Networks import cnn_2layer_fc_model_mnist, cnn_3layer_fc_model_mnist, mnist_student,train_models


def parseArg():
    parser = argparse.ArgumentParser(description='FedMD, a federated learning framework. \
    Participants are training collaboratively. ')
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for FedMD.'
                        )

    conf_file = os.path.abspath("conf/EMNIST_conf.json")

    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layer_fc_model_mnist,
                    "3_layer_CNN": cnn_3layer_fc_model_mnist}

student_model = mnist_student

if __name__ == "__main__":
    conf_file = parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())

        # n_classes = conf_dict["n_classes"]
        model_config = conf_dict["models"]
        pre_train_params = conf_dict["pre_train_params"]
        model_saved_dir = conf_dict["model_saved_dir"]
        model_saved_names = conf_dict["model_saved_names"]
        is_early_stopping = conf_dict["early_stopping"]
        public_classes = conf_dict["public_classes"]
        private_classes = conf_dict["private_classes"]
        n_epochs = conf_dict["epochs"]
        n_classes = len(public_classes) + len(private_classes)
        alpha = conf_dict["alpha"]
        manualseed = conf_dict["manualseed"]

        emnist_data_dir = conf_dict["EMNIST_dir"]
        N_parties = conf_dict["N_parties"]
        N_samples_per_class = conf_dict["N_samples_per_class"]

        checkpoint = conf_dict["checkpoint"]
        N_rounds = conf_dict["N_rounds"]
        N_alignment = conf_dict["N_alignment"]
        N_private_training_round = conf_dict["N_private_training_round"]
        private_training_batchsize = conf_dict["private_training_batchsize"]
        N_logits_matching_round = conf_dict["N_logits_matching_round"]
        logits_matching_batchsize = conf_dict["logits_matching_batchsize"]

        result_save_dir = conf_dict["result_save_dir"]

    del conf_dict, conf_file

    X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST \
        = load_MNIST_data(standarized=True, verbose=True)

    public_dataset = {"X": X_train_MNIST, "y": y_train_MNIST}

    X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST, writer_ids_train, writer_ids_test \
        = load_EMNIST_data(emnist_data_dir,
                           standarized=True, verbose=True)

    print(np.unique(y_train_EMNIST))

    X_train_EMNIST, y_train_EMNIST \
        = generate_partial_data(X=X_train_EMNIST, y=y_train_EMNIST,
                                class_in_use=private_classes,
                                verbose=True)

    X_test_EMNIST, y_test_EMNIST \
        = generate_partial_data(X=X_test_EMNIST, y=y_test_EMNIST,
                                class_in_use=private_classes,
                                verbose=True)

    for index, cls_ in enumerate(private_classes):
        y_train_EMNIST[y_train_EMNIST == cls_] = index + len(public_classes)
        y_test_EMNIST[y_test_EMNIST == cls_] = index + len(public_classes)
    del index, cls_


    mod_private_classes = np.arange(len(private_classes)) + len(public_classes)  # [10,11,12,13,14,15]

    # generate private data
    private_data, total_private_data, source_data, total_source_data \
        = generate_dirc_private_data(X_train_EMNIST, y_train_EMNIST,
                                     alpha=alpha,
                                     N_parties=N_parties,
                                     classes_in_use=mod_private_classes,
                                     num_source_samples=100,
                                     manualseed=manualseed,
                                     size_limit=True)



    X_tmp, y_tmp = generate_partial_data(X=X_test_EMNIST, y=y_test_EMNIST,
                                         class_in_use=mod_private_classes, verbose=True)
    private_test_data = {"X": X_tmp, "y": y_tmp}
    del X_tmp, y_tmp

    # convert dataset to train_loader and test_loader
    train_dataset = (TensorDataset(torch.from_numpy(X_train_MNIST).float(), torch.from_numpy(y_train_MNIST).long()))
    test_dataset = (TensorDataset(torch.from_numpy(X_test_MNIST).float(), torch.from_numpy(y_test_MNIST).long()))

    pre_models_dir = "./pretrained_MNIST/"

    parties = []

    # for i, item in enumerate(model_config):
    #     model_name = item["model_type"]
    #     model_params = item["params"]
    #     tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes,**model_params)
    #     print("model {0} : {1}".format(i, model_saved_names[i]))
    #     print(tmp)
    #     parties.append(tmp)
    #
    #     del model_name, model_params, tmp
    #     #END FOR LOOP
    # pre_train_result = train_models(parties,train_dataset,test_dataset,num_epochs=n_epochs,save_dir = pre_models_dir, save_names = model_saved_names)

    # In fedmd, each model is first trained on the public dataset, i.e., mnist dataset.
    # To save the experiment time, we provide pre_trained local models in the the "./pretrained_MNIST/" folder
    # If someone wants to conduct experiments from scratch, please use the above code to obtain the pretrained model

    for i, item in enumerate(model_config):
        model_name = item["model_type"]
        model_params = item["params"]
        tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes, **model_params)
        print("model {0} : {1}".format(i, model_saved_names[i]))
        print(tmp)
        tmp.load_state_dict(torch.load(os.path.join(pre_models_dir, "{}.h5".format(model_saved_names[i]))))
        parties.append(tmp)

        del model_name, model_params, tmp

    student_model = student_model(num_classes=n_classes)

    del X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST, \
        X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST, writer_ids_train, writer_ids_test

    fedmd = FedMD_SIAs(parties,
                       public_dataset=public_dataset,
                       private_data=private_data,
                       s_model=student_model,
                       alpha=alpha,
                       model_saved_name=model_saved_names,
                       total_private_data=total_private_data,
                       private_test_data=private_test_data,
                       source_data=source_data,
                       manualseed=manualseed,
                       checkpoint=checkpoint,
                       total_source_data=total_source_data,
                       N_rounds=N_rounds,
                       N_alignment=N_alignment,
                       N_logits_matching_round=N_logits_matching_round,
                       logits_matching_batchsize=logits_matching_batchsize,
                       N_private_training_round=N_private_training_round,
                       private_training_batchsize=private_training_batchsize)

    initialization_result = fedmd.init_result

    collaboration_performance = fedmd.collaborative_training_SIA()
