import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import functools
import operator
from torch.utils.data import TensorDataset, DataLoader,Dataset
from .utility import RunningAverage,set_logger

from torch.optim import SGD, Adam
import os

class CNN(nn.Module):
    """Basic Pytorch CNN implementation"""

    def __init__(self, in_channels, out_channels, input_dim=(3,32,32)):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=100),
            nn.Linear(in_features=100, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)

        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out

class cifar_student(nn.Module):
    def __init__(self, num_classes):
        super(cifar_student, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class mnist_student(nn.Module):
    def __init__(self, num_classes):
        super(mnist_student, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class cnn_3layer_fc_model_cifar(nn.Module):

    def __init__(self, n_classes,n1 = 128, n2=192, n3=256, dropout_rate = 0.2,input_dim=(3,32,32)):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=(3,3), stride=1),
            nn.BatchNorm2d(n1,momentum=0.99, eps=0.001),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=(2,2),stride=1),

            nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=(2,2), stride=2),
            nn.BatchNorm2d(n2,momentum=0.99, eps=0.001),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=(2,2),stride=2),

            nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=(3,3), stride=2),
            nn.BatchNorm2d(n3,momentum=0.99, eps=0.001),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)  # flatten the vector
        out = self.classifier(out)
        return out

class cnn_3layer_fc_model_mnist(nn.Module):

    def __init__(self, n_classes,n1 = 128, n2=192, n3=256, dropout_rate = 0.2,input_dim=(1,28,28)):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n1, kernel_size=(3,3), stride=1),
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=(2,2),stride=1),

            nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=(2,2), stride=2),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=(2,2),stride=2),

            nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=(3,3), stride=2),
            nn.BatchNorm2d(n3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)  # flatten the vector
        out = self.classifier(out)
        return out

class cnn_2layer_fc_model_cifar(nn.Module):

    def __init__(self, n_classes,n1 = 128, n2=256, dropout_rate = 0.2,input_dim=(3,32,32)):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=(3,3), stride=1),
            nn.BatchNorm2d(n1,momentum=0.99, eps=0.001),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=(2,2),stride=1),

            nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=(3,3), stride=2),
            nn.BatchNorm2d(n2,momentum=0.99, eps=0.001),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # nn.AvgPool2d(kernel_size=(2,2),stride=2),
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)  # flatten the vector
        out = self.classifier(out)
        return out

class cnn_2layer_fc_model_mnist(nn.Module):

    def __init__(self, n_classes,n1 = 128, n2=192, dropout_rate = 0.2,input_dim=(1,28,28)):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n1, kernel_size=(3,3), stride=1),
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=(2,2),stride=1),

            nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=(2,2), stride=2),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)  # flatten the vector
        out = self.classifier(out)
        return out

# ************************** training function **************************
def train_epoch(model, data_loader, cuda=True, lr=0.001,batch_size=128,loss_fn = nn.CrossEntropyLoss(),weight_decay=1e-3):
    optim = Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    # optim = SGD(model.parameters(), lr=0.001)
    loss_fn = loss_fn

    if cuda:
        device = torch.device("cuda:0")
        model.to(device)

    model.train()
    loss_avg = RunningAverage()

    data_loader=DataLoader(data_loader,batch_size=batch_size,shuffle=True)

    with tqdm(total=len(data_loader),disable=True) as t:  # Use tqdm for progress bar

        for i, (train_batch, labels_batch) in enumerate(data_loader):

            if cuda:
                train_batch = train_batch.cuda()        # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output and loss
            output_batch = model(train_batch)           # logit without softmax
            loss = loss_fn(output_batch, labels_batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the average loss
            loss_avg.update(loss.item())

            # tqdm setting
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

def train(model, data_loader, epochs, cuda=True, lr=0.001,batch_size=128,loss_fn = nn.CrossEntropyLoss(),weight_decay=1e-3):
    for epoch in range(epochs):
        # ********************* one full pass over the training set *********************
        train_epoch(model, data_loader, lr=lr,cuda=cuda, batch_size=batch_size,loss_fn = loss_fn,weight_decay=weight_decay)
    return model

def evaluate(model, data_loader, cuda=True):
    loss_fn = nn.CrossEntropyLoss()
    data_loader=DataLoader(data_loader,batch_size=128,shuffle=False)
    model.eval()
    # summary for current eval loop
    summ = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            if cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    return metrics_mean

def train_and_eval(model, train_loader, dev_loader, num_epochs,batch_size,lr=0.001,weight_decay=0.001):


    model_trained=train(model,train_loader,epochs=num_epochs,lr=lr,batch_size=batch_size,weight_decay=weight_decay)

    # ********************* Evaluate for one epoch on training and validation set *********************
    val_metrics = evaluate(model_trained, dev_loader, cuda=True)     # {'acc':acc, 'loss':loss}
    train_metrics = evaluate(model_trained, train_loader, cuda=True)     # {'acc':acc, 'loss':loss}

    val_acc = val_metrics['acc']
    val_loss=val_metrics['loss']
    train_acc = train_metrics['acc']
    train_loss=train_metrics['loss']

    print('Val acc:',val_acc)
    print('Val loss:',val_loss)
    print('Train acc:',train_acc)
    print('Train loss:',train_loss)
    return model_trained, train_acc,train_loss,val_acc,val_loss


def train_models(models, train_loader, dev_loader,num_epochs,batch_size = 128, save_dir = "./", save_names = None):
    resulting_val_acc = []
    record_result = []
    for n, model in enumerate(models):
        print("Training model ", n)
        model,train_acc,train_loss,val_acc,val_loss=train_and_eval(model,train_loader,dev_loader,num_epochs,batch_size=batch_size)



        # save_name = os.path.join(save_path, 'model_{}.tar'.format(n))

        resulting_val_acc.append(val_acc)
        record_result.append({"train_acc": train_acc,
                              "val_acc": val_acc,
                              "train_loss": train_loss,
                              "val_loss": val_loss})

        file_name = save_names[n] + ".h5"
        torch.save(model.state_dict(),os.path.join(save_dir,file_name))

    print("pre-train accuracy:")
    print(resulting_val_acc)

    return record_result
