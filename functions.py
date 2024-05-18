import torch
from torch import nn
import torch.utils.data
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix

def accuracy_fn(y_true, pred):
    correct = torch.eq(y_true, pred).sum().item()
    acc = (correct / len(pred)) * 100
    return acc

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader) #find loss per batch
    train_acc /= len(data_loader)
    print(f"\nTrain loss: {train_loss:.5f} | Train acc: {train_acc:.2f}\n")

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X_test, y_test in data_loader:
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, pred=test_pred.argmax(dim=1))
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    print(f"\nTest loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

def confusion_matrix(model: torch.nn.Module,
                          data_loader: torch.utils.data.DataLoader,
                          num_class: int,
                          data: list):
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in (data_loader):
        #add a batch dim (to match shape)
            y_logits = model(X)
            y_pred = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1) 
            y_preds.append(y_pred)

    y_pred_tensor = torch.cat(y_preds)

    confmat = ConfusionMatrix(num_classes=len(num_class), task='multiclass')
    confmat_tensor = confmat(preds=y_pred_tensor,
                         target=data.targets)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=num_class, 
        figsize=(10, 7)
    )
    plt.show()