import numpy as np
import copy
from torch.utils.data import TensorDataset, DataLoader
import torch
from utils.data_utils import generate_alignment_data
from utils.Neural_Networks import train_and_eval, evaluate, train
import torch.nn as nn
from utils.Sia import SIA
from utils.logger import Logger, mkdir_p
import os


def get_logits(model, data_loader, cuda=True):
    model.eval()
    # logits for of the unlabeld public dataset
    logits = []
    data_loader = DataLoader(data_loader, batch_size=128, shuffle=False)
    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch in data_loader:

            if cuda:
                data_batch = data_batch.cuda()  # (B,3,32,32)

            # compute model output
            output_batch = model(data_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()

            # append logits
            logits.append(output_batch)

    # get 2-D logits array
    logits = np.concatenate(logits)

    return logits


class FedMD_SIAs():
    def __init__(self, parties, s_model, public_dataset,
                 private_data, total_private_data,
                 private_test_data, source_data, total_source_data, N_alignment,
                 N_rounds,
                 alpha,
                 manualseed,
                 checkpoint,
                 model_saved_name,
                 N_logits_matching_round, logits_matching_batchsize,
                 N_private_training_round, private_training_batchsize):
        self.student_model = s_model
        self.alpha = alpha
        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.model_saved_name = model_saved_name
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.total_private_data = total_private_data
        self.source_data = source_data
        self.checkpoint = checkpoint
        self.manualseed = manualseed
        self.total_source_data = total_source_data
        self.N_alignment = N_alignment

        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize

        self.collaborative_parties = []
        self.init_result = []

        print("start model initialization: ")
        test_dataset = (TensorDataset(torch.from_numpy(private_test_data["X"]).float(),
                                      torch.from_numpy(private_test_data["y"]).long()))
        for i in range(self.N_parties):
            print("model ", i)
            model_A_twin = copy.deepcopy(parties[i])

            print("start full stack training ... ")
            # get train_loader and test_loader
            train_dataset = (TensorDataset(torch.from_numpy(private_data[i]["X"]).float(),
                                           torch.from_numpy(private_data[i]["y"]).long()))

            model_A, train_acc, train_loss, val_acc, val_loss = train_and_eval(model_A_twin, train_dataset,
                                                                               test_dataset, 25, batch_size=32)

            print("full stack training done")

            self.collaborative_parties.append({"model": model_A})

            self.init_result.append({"val_acc": val_acc,
                                     "train_acc": train_acc,
                                     "val_loss": val_loss,
                                     "train_loss": train_loss,
                                     })

            del model_A, model_A_twin

        # END FOR LOOP

    def collaborative_training_SIA(self):
        # start collaborating training

        if not os.path.isdir(self.checkpoint):
            mkdir_p(self.checkpoint)

        logger = Logger(os.path.join(self.checkpoint, 'log_seed{}.txt'.format(self.manualseed)))
        logger.set_names(['alpha', 'comm. round', 'ASR'])

        test_dataset = (TensorDataset(torch.from_numpy(self.private_test_data["X"]).float(),
                                      torch.from_numpy(self.private_test_data["y"]).long()))
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        best_att_asr = 0
        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            alignment_dataloader = (torch.from_numpy(alignment_data["X"]).float())
            print("round ", r)

            print("update logits ... ")
            print("start source inference attacks:")
            mimic_local_models = []

            for index, d in enumerate(self.collaborative_parties):
                train_dataset = (TensorDataset(torch.from_numpy(self.private_data[index]["X"]).float(),
                                               torch.from_numpy(self.private_data[index]["y"]).long()))
                student_model = copy.deepcopy(self.student_model)
                logit_currently = get_logits(d["model"], alignment_dataloader, cuda=True)
                public_dataloader = (TensorDataset(torch.from_numpy(alignment_data["X"]).float(),
                                                   torch.from_numpy(logit_currently).float()))

                student_model = train(student_model, public_dataloader, lr=0.001, epochs=30, cuda=True, batch_size=64,
                                      loss_fn=nn.L1Loss(), weight_decay=0)
                mimic_local_models.append(student_model)

            SIAs = SIA(w_locals=mimic_local_models, dataset=self.source_data)
            attack_ASR = SIAs.attack()

            logger.append([self.alpha, r, attack_ASR])

            if attack_ASR > best_att_asr:
                for i, d in enumerate(self.collaborative_parties):
                    file_name = self.model_saved_name[i] + ".h5"
                    torch.save(d["model"].state_dict(), os.path.join(self.checkpoint, file_name))

            best_att_asr = max(best_att_asr, attack_ASR)

            # update logits
            logits = 0

            for d in self.collaborative_parties:
                logits += get_logits(d["model"], alignment_dataloader, cuda=True)

            logits /= self.N_parties

            # test performance
            print("test performance ... ")

            private_test_dataloader = (TensorDataset(torch.from_numpy(self.private_test_data["X"]).float(),
                                                     torch.from_numpy(self.private_test_data["y"]).long()))

            for index, d in enumerate(self.collaborative_parties):
                metrics_mean = evaluate(d["model"], private_test_dataloader, cuda=True)
                collaboration_performance[index].append(metrics_mean["acc"])

                print(collaboration_performance[index][-1])

            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                public_dataloader = (
                    TensorDataset(torch.from_numpy(alignment_data["X"]).float(), torch.from_numpy(logits).float()))

                model_alignment = train(d["model"], public_dataloader, epochs=self.N_logits_matching_round, cuda=True,
                                        batch_size=self.logits_matching_batchsize,
                                        loss_fn=nn.L1Loss())

                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))

                private_dataloader = (TensorDataset(torch.from_numpy(self.private_data[index]["X"]).float(),
                                                    torch.from_numpy(self.private_data[index]["y"]).long()))
                model_local = train(model_alignment, private_dataloader, epochs=self.N_private_training_round,
                                    cuda=True, batch_size=self.private_training_batchsize,
                                    loss_fn=nn.CrossEntropyLoss())

                d["model"] = model_local
                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        return collaboration_performance
