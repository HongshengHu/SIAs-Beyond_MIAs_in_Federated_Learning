import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import logging

def show_dataset_samples(classes, samples_per_class, 
                         images, labels, data_type="MNIST"):
    num_classes = len(classes)
    fig, axes = plt.subplots(samples_per_class, num_classes, 
                             figsize=(num_classes, samples_per_class)
                            )
    
    for col_index, cls in enumerate(classes):
        idxs = np.flatnonzero(labels == cls)
        idxs = np.random.choice(idxs, samples_per_class, 
                                replace=False)
        for row_index, idx in enumerate(idxs):    
            if data_type == "MNIST":
                axes[row_index][col_index].imshow(images[idx],
                                                  cmap = 'binary', 
                                                  interpolation="nearest")
                axes[row_index][col_index].axis("off")
            elif data_type == "CIFAR":
                axes[row_index][col_index].imshow(images[idx].astype('uint8'))
                axes[row_index][col_index].axis("off")
                
            else:
                print("Unknown Data type. Unable to plot.")
                return None
            if row_index==0:
                axes[row_index][col_index].set_title("Class {0}".format(cls))
                
                
    plt.show()
    return None


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# def plot_history(model):
    
#     """
#     input : model is trained keras model.
#     """
    
#     fig, axes = plt.subplots(2,1, figsize = (12, 6), sharex = True)
#     axes[0].plot(model.history.history["loss"], "b.-", label = "Training Loss")
#     axes[0].plot(model.history.history["val_loss"], "k^-", label = "Val Loss")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Loss")
#     axes[0].legend()
    
    
#     axes[1].plot(model.history.history["acc"], "b.-", label = "Training Acc")
#     axes[1].plot(model.history.history["val_acc"], "k^-", label = "Val Acc")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("Accuracy")
#     axes[1].legend()
    
#     plt.subplots_adjust(hspace=0)
#     plt.show()
    
# def show_performance(model, Xtrain, ytrain, Xtest, ytest):
#     y_pred = None
#     print("CNN+fC Training Accuracy :")
#     y_pred = model.predict(Xtrain, verbose = 0).argmax(axis = 1)
#     print((y_pred == ytrain).mean())
#     print("CNN+fc Test Accuracy :")
#     y_pred = model.predict(Xtest, verbose = 0).argmax(axis = 1)
#     print((y_pred == ytest).mean())
#     print("Confusion_matrix : ")
#     print(confusion_matrix(y_true = ytest, y_pred = y_pred))
    
#     del y_pred