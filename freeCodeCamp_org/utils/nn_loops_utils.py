from tqdm.auto import tqdm
from colorama import init
from typing import Callable
import torch

init()
from colorama import Fore, Back, Style


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.modules.loss,
               accuracy_fn: Callable,
               device: str = 'cuda',
               tqdm_object=None,
               print_end='\n'
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
            tqdm_object.set_postfix_str(batch_log_string)
        else:
            print(batch_log_string)

    loss /= len(data_loader)
    accuracy /= len(data_loader)

    epoch_log_string = Fore.GREEN + f'Train Loss: {loss:.4f} | Train Accuracy {accuracy:.4f}' + Fore.RESET
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

    log_string = Fore.BLUE + f'Test Loss:  {loss:.4f} | Test Accuracy  {accuracy:.4f}' + Fore.RESET
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


def make_predictions(model: torch.nn.Module, data: list, device: torch.device = 'cuda'):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device)  # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(),
                                      dim=0)  # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)
