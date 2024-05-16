import torch
from torch import nn
import torch.utils
from torch.utils.data import DataLoader


def accuracy_fn(y_true, pred):
    correct = torch.eq(y_true, pred).sum().item()
    acc = (correct / len(pred)) * 100
    return acc

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn
               ):
    train_loss, train_acc = 0
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
    test_loss, test_acc = 0
    model.eval()

    with torch.inference_mode():
        for X_test, y_test in data_loader:
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    print(f"\nTest loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")