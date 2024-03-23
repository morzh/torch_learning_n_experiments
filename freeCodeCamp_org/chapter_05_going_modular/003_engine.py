"""
Contains functions for training and testing a PyTorch model.
"""
import sys
import torch

from tqdm.auto import tqdm
from colorama import init

init()
from colorama import Fore, Back, Style


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          number_epochs: int = 10,
          device: str = 'cuda',
          model_filename: str = 'model.pth',
          use_check_point: bool = True,
          save_best_model: bool = True,
          tqdm_bar_width: int = 50) -> dict[str, list]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_accuracy: [...],
              test_loss: [...],
              test_accuracy: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_accuracy: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_accuracy: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_accuracy": [],
               "test_loss": [],
               "test_accuracy": []
               }

    model.to(device)
    if use_check_point:
        print(f'Loading weights from {model_filename}')
        model.load_state_dict(torch.load(model_filename))

    tqdm_loop_object = tqdm(range(number_epochs),
                            bar_format=f'{{l_bar}}{{bar:{tqdm_bar_width}}}{{r_bar}}{{bar:-{tqdm_bar_width}b}}',
                            position=0,
                            leave=False,
                            file=sys.stdout)

    if use_check_point:
        _, test_accuracy_previous = test_step(model=model,
                                              data_loader=test_dataloader,
                                              loss_fn=loss_fn,
                                              device=device,
                                              tqdm_object=tqdm_loop_object, )
    else:
        test_accuracy_previous = 0.0

    for epoch in tqdm_loop_object:
        tqdm_loop_object.write(Style.BRIGHT + Fore.CYAN + f'Epoch {epoch}' + Fore.RESET + Style.RESET_ALL)
        train_loss, train_accuracy = train_step(model=model,
                                                data_loader=train_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                device=device,
                                                tqdm_object=tqdm_loop_object,
                                                )
        test_loss, test_accuracy = test_step(model=model,
                                             data_loader=test_dataloader,
                                             loss_fn=loss_fn,
                                             device=device,
                                             tqdm_object=tqdm_loop_object, )

        if test_accuracy > test_accuracy_previous and save_best_model:
            torch.save(model.state_dict(), f=model_filename)
        test_accuracy_previous = test_accuracy

        # Print out what's happening
        # print(
        #     f"Epoch: {epoch + 1} | "
        #     f"train_loss: {train_loss:.4f} | "
        #     f"train_accuracy: {train_accuracy:.4f} | "
        #     f"test_loss: {test_loss:.4f} | "
        #     f"test_accuracy: {test_accuracy:.4f}"
        # )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)

    return results


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str = 'cuda',
               tqdm_object=None,
               ) -> tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_accuracy = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_predicted = model(X)
        loss = loss_fn(y_predicted, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        # y_predicted_classes = torch.argmax(torch.softmax(y_predicted, dim=1), dim=1)
        y_predicted_classes = torch.argmax(y_predicted, 1)
        train_accuracy += (y_predicted_classes == y).sum().item() / len(y_predicted_classes)

        batch_log_string = f'Train batch {batch} of {len(data_loader)}'
        if tqdm_object is not None:
            tqdm_object.set_postfix_str(batch_log_string)
        else:
            print(batch_log_string)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)

    tqdm_string = Fore.GREEN + f'Train Loss:  {train_loss:.4f} | Train Accuracy  {train_accuracy:.4f}' + Fore.RESET
    if tqdm_object is not None:
        tqdm_object.write(tqdm_string)
    else:
        print(tqdm_string)

    return train_loss, train_accuracy


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str = 'cuda',
              tqdm_object=None,
              ) -> tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    model.eval()
    test_loss, test_accuracy = 0.0, 0.0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            test_predictions_logits = model(X)
            loss = loss_fn(test_predictions_logits, y)
            test_loss += loss.item()
            test_prediction_labels = test_predictions_logits.argmax(dim=1)
            test_accuracy += ((test_prediction_labels == y).sum().item() / len(test_prediction_labels))

            batch_log_string = f'Test batch {batch} of {len(data_loader)}'
            if tqdm_object is not None:
                tqdm_object.set_postfix_str(batch_log_string)
            else:
                print(batch_log_string)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss /= len(data_loader)
    test_accuracy /= len(data_loader)

    log_string = Fore.BLUE + f'Test Loss:  {loss:.4f} | Test Accuracy  {test_accuracy:.4f}' + Fore.RESET
    if tqdm_object is not None:
        tqdm_object.write(log_string)
    else:
        print(log_string)

    return test_loss, test_accuracy
