from typing import Callable
import sys

import torch
from torch import nn
import torchmetrics
from tqdm.auto import tqdm

from fashionMnist_datasets import *


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.modules.loss,
               accuracy_fn: Callable,
               device: str = 'cuda',
               tqdm_object=None,
               ):
    """
    Perform training with model on DataLoader
    :param model:
    :param data_loader:
    :param optimizer:
    :param loss_fn:
    :param accuracy_fn:
    :param device:
    :param tqdm_object:
    """

    loss, accuracy = 0, 0
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        # X = X / 255
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        batch_loss = loss_fn(y_pred, y)
        loss += batch_loss
        accuracy += accuracy_fn(y_pred.argmax(dim=1), y)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_log_string = f'Batch {batch} of {len(data_loader)}'
        if tqdm_object is not None:
            tqdm_object.write(batch_log_string)
        else:
            print(batch_log_string)

    loss /= len(data_loader)
    accuracy /= len(data_loader)

    epoch_log_string = f'Train Loss: {loss:.4f} | Train Accuracy {test_accuracy:.4f}'
    if tqdm_object is not None:
        tqdm_object.write(epoch_log_string)
    else:
        print(epoch_log_string)


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.modules.loss,
              accuracy_fn: Callable,
              device: str = 'cuda',
              tqdm_object=None,
              ):
    """
    Perform training with model on DataLoader
    :param model:
    :param data_loader:
    :param accuracy_fn:
    :param device:
    :param tqdm_object:
    :return:
    """

    loss, accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y)
            accuracy += accuracy_fn(pred.argmax(dim=1), y)

        loss /= len(data_loader)
        accuracy /= len(data_loader)

    log_string = f'Train Loss: {loss:.4f} Test Accuracy {accuracy:.4f}'
    if tqdm_object is not None:
        tqdm_object.write(log_string)
    else:
        print(log_string)


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn: Callable,
               device: str = 'cuda',
               ):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0.0, 0.0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_pred.argmax(dim=1), y)

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


input_shape = 28 * 28
hidden_units = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
print('torch version:', torch.__version__, 'device:', device)


class FashionMNNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            # nn.Softplus(),
        )

    def forward(self, x):
        return self.layers_stack(x)


model = FashionMNNISTModelV1(
    input_shape=input_shape,
    hidden_units=hidden_units,
    output_shape=len(class_names)
).to(device)

print(model)

number_epochs = 250
number_classes = len(test_data.classes)
loss_function = nn.CrossEntropyLoss().to(device)
accuracy_function = torchmetrics.Accuracy(num_classes=number_classes, task='multiclass').to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.08, )
# optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.1,)


torch.manual_seed(42)
tqdm_bar_width = 60
tqdm_loop_object = tqdm(range(number_epochs),
                        bar_format=f'{{l_bar}}{{bar:{tqdm_bar_width}}}{{r_bar}}{{bar:-{tqdm_bar_width}b}}',
                        position=0,
                        leave=False,
                        file=sys.stdout)

for epoch in tqdm_loop_object:
    print(f'Epoch: {epoch} \n---------------')
    train_step(model=model,
               data_loader=train_loader,
               loss_fn=loss_function,
               accuracy_fn=accuracy_function,
               optimizer=optimizer,
               device=device,
               tqdm_object=tqdm_loop_object,
               )

    test_step(model=model,
              data_loader=test_loader,
              loss_fn=loss_function,
              accuracy_fn=accuracy_function,
              device=device,
              tqdm_object=tqdm_loop_object,
              )

