import numpy as np
import matplotlib.pyplot as plt

# TODO: 채우기
class LossTracker(object):
    def __init__(self):
        self.legend = ['train']
        self.loss = []

    def add_train_loss(self, train_loss):
        self.loss.append(train_loss)

    def plot_loss(self, train_loss_list):
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.log(train_loss_list))
        ax.set_title("Training Loss", fontsize=30)
        ax.set_xlabel("Epoch", fontsize=30)
        ax.set_ylabel("Loss (log)", fontsize=30)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid()
        
        ax.legend(self.legend, loc='upper right', fontsize=20)
        plt.tight_layout()

    def save_figure(self, name):
        self.plot_loss(self.loss)
        plt.savefig(name)
        plt.close('all')