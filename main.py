import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

#visualization
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix


training_data = datasets.MNIST(root="numbers",
                               download=True,
                               train=True,
                               transform=ToTensor,
                               target_transform=None)

testing_data = datasets.MNIST(root="numbers",
                              download=True,
                              train=False,
                              transform=ToTensor,
                              target_transform=None)

number_labels = training_data.classes

#into batches rather than each image
BATCH_SIZE = 32
train_dataLoader = DataLoader(dataset=training_data,
                              shuffle=True,
                              batch_size=BATCH_SIZE,
                              )

test_dataLoader = DataLoader(dataset=testing_data,
                             shuffle=False,
                             batch_size=BATCH_SIZE)



