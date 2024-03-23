import sys
from tqdm.auto import tqdm
import torch
from torch import nn
import torchmetrics
from fashionMnist_datasets import *

sys.path.insert(0, "../utils")
from nn_loops_utils import train_step, test_step, eval_model

from colorama import init
init()
from colorama import Fore, Back, Style

from torchmetrics import ConfusionMatrix
import mlxtend
from mlxtend.plotting import plot_confusion_matrix

'''
CNN Explainer
https://poloclub.github.io/cnn-explainer/
'''

do_training = True
do_plot_confusion_matrix = True

model_filename = 'tiny_vgg_v002.pth'


class TinyVGG(nn.Module):
    def __init__(self, input_channels: int, hidden_units: int, number_classes: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=hidden_units*7*7),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units*7*7, out_features=number_classes)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_units = 64+32
number_classes = len(test_data.classes)
number_epochs = 180
tqdm_bar_width = 50

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = TinyVGG(input_channels=1, hidden_units=10, number_classes=number_classes).to(device)
loss_function = nn.CrossEntropyLoss()
accuracy_function = torchmetrics.Accuracy(num_classes=number_classes, task='multiclass').to(device)
print(model)

if do_training:
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.10)
    tqdm_loop_object = tqdm(range(number_epochs),
                            bar_format=f'{{l_bar}}{{bar:{tqdm_bar_width}}}{{r_bar}}{{bar:-{tqdm_bar_width}b}}',
                            position=0,
                            leave=False,
                            file=sys.stdout)

    for epoch in tqdm_loop_object:
        tqdm_loop_object.write(Style.BRIGHT + Fore.CYAN + f'Epoch {epoch}' + Fore.RESET + Style.RESET_ALL)
        train_step(model=model,
                   data_loader=train_loader,
                   loss_fn=loss_function,
                   accuracy_fn=accuracy_function,
                   optimizer=optimizer,
                   device=device,
                   tqdm_object=tqdm_loop_object,
                   print_end=' '
                   )

        test_step(model=model,
                  data_loader=test_loader,
                  loss_fn=loss_function,
                  accuracy_fn=accuracy_function,
                  device=device,
                  tqdm_object=tqdm_loop_object,
                  )

        if epoch > 25:
            optimizer.param_groups[0]['lr'] *= 0.5
        elif epoch > 60:
            optimizer.param_groups[0]['lr'] *= 0.4
        elif epoch > 100:
            optimizer.param_groups[0]['lr'] *= 0.3

    model_results = eval_model(model=model,
                               data_loader=test_loader,
                               loss_fn=loss_function,
                               accuracy_fn=accuracy_function,
                               device=device)

    print(model_results)
    torch.save(obj=model.state_dict(), f=model_filename)


if do_plot_confusion_matrix:
    print('mlxtend version', mlxtend.__version__)
    model.load_state_dict(torch.load(model_filename))
    model.to(device)
    model.eval()

    loaded_model_results = eval_model(model=model,
                                      data_loader=test_loader,
                                      loss_fn=loss_function,
                                      accuracy_fn=accuracy_function)
    print(loaded_model_results)

    y_preds = []
    with torch.inference_mode():
        for X, y in tqdm(test_loader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            y_preds.append(y_pred.cpu())
    y_pred_tensor = torch.cat(y_preds)

    confusion_matrix = ConfusionMatrix(num_classes=number_classes, task='multiclass')
    confusion_tensor = confusion_matrix(preds=y_pred_tensor, target=test_data.targets)

    figure, axis = plot_confusion_matrix(
        conf_mat=confusion_tensor.numpy(),
        class_names=class_names,
        figsize=(10, 7),
    )
    plt.show()

