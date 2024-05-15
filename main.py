import torch
from torch import nn
from model import NumberModel
from load import *

#visualization
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix

num_model = NumberModel(input_shape=1,
                        hidden_units=10,
                        output_shape=len(number_labels))






