import torch
from torch import nn
from model import NumberModel
from load import *
from functions import *
import matplotlib.pyplot as plt


num_model = NumberModel(input_shape=1,
                        hidden_units=10,
                        output_shape=len(number_labels))

#learning rate
LR = 0.1
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(num_model.parameters(), lr=LR)

epochs = 5
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    train_step(model=num_model,
               data_loader=train_dataLoader,
               loss_fn=loss_function,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn)
    
    test_step(model=num_model,
              data_loader=test_dataLoader,
              loss_fn=loss_function,
              accuracy_fn=accuracy_fn)

##visualize what it's getting wrong
confusion_matrix(model=num_model,
                 data_loader=test_dataLoader,
                 num_class=number_labels,
                 data=testing_data)