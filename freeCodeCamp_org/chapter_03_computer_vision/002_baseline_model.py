import sys
from fashionMnist_datasets import *
import torch
from torch import nn
import torchmetrics
from tqdm.auto import tqdm


input_shape = 28*28
hidden_units = 128


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
print('torch version:', torch.__version__, 'device:', device)


class FashionMNNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            # nn.Dropout(0.1),
            # nn.Softplus(beta=20),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.Dropout(0.1),
            # nn.Softplus(beta=20),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            # nn.Softplus(beta=5),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.layers_stack(x)


model_0 = FashionMNNISTModelV0(
    input_shape=input_shape,
    hidden_units=hidden_units,
    output_shape=len(class_names)
).to(device)

print(model_0)

number_epochs = 250
number_classes = len(test_data.classes)
loss_function = nn.CrossEntropyLoss().to(device)
accuracy = torchmetrics.Accuracy(num_classes=number_classes, task='multiclass').to(device)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.08,)
# optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.1,)


torch.manual_seed(42)
tqdm_bar_width = 60
tqdm_loop_object = tqdm(range(number_epochs),
                        bar_format=f'{{l_bar}}{{bar:{tqdm_bar_width}}}{{r_bar}}{{bar:-{tqdm_bar_width}b}}',
                        position=0,
                        leave=False,
                        file=sys.stdout)

for epoch in tqdm_loop_object:
    train_loss = 0
    for batch, (X, y) in enumerate(train_loader):
        model_0.train()
        X = X.to(device)
        y = y.to(device)
        y_pred = model_0(X)
        loss = loss_function(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            tqdm_loop_object.set_postfix_str(f'Epoch {epoch}, batch {batch} of {len(train_loader)}')

    if epoch > 17:
        optimizer.param_groups[0]['lr'] = 0.02
    elif epoch > 42:
        optimizer.param_groups[0]['lr'] = 0.005

    train_loss /= len(train_loader)
    test_loss, test_accuracy = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            test_pred = model_0(X_test)
            test_loss += loss_function(test_pred, y_test)
            test_accuracy += accuracy(test_pred.argmax(dim=1), y_test)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

    tqdm_loop_object.write(
        f'Train Loss: {train_loss:.4f} | Test Loss {test_loss:.4f}| Test Accuracy {test_accuracy:.4f}',
    )

print(next(model_0.parameters()).device)
