import sklearn
from sklearn.datasets import make_circles
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch, torchvision
from torch import nn
import requests
from pathlib import Path


def download_helper_functions():
    if Path("helper_functions.py").is_file():
        print("helper_functions.py already exists, skipping download")
    else:
        print("Downloading helper_functions.py")
        request = requests.get(
            "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
        with open("helper_functions.py", "wb") as f:
            f.write(request.content)


def decision_boundary(model, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()


download_helper_functions()
from helper_functions import plot_predictions, plot_decision_boundary


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch version:', torch.__version__, 'device:', device)

n_samples = 50000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# print(X.shape, y.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=matplotlib.colormaps['RdYlBu'])
plt.show()

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        # self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        # x = self.relu(self.layer_2(x))
        return self.layer_3(x)


model = CircleModelV2().to(device)


loss_function = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(params=model.parameters(), lr=0.25)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.08)

# print(model_0.state_dict())


def accuracy(y_true, y_pred):
    matches = torch.eq(y_true, y_pred).sum().item()
    accuracy_percents = matches / len(y_pred) * 100
    return accuracy_percents


torch.manual_seed(42)
torch.cuda.manual_seed(42)

number_epochs = 700
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


for epoch in range(number_epochs):
    model.train()

    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_function(y_logits, y_train)
    acc = accuracy(y_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_function(test_logits, y_test)
        test_accuracy = accuracy(y_test, test_preds)

        if epoch % 100 == 0:
            print(f'Epoch: {epoch} | Loss {test_loss:.5f}, Accuracy: {acc:.2f}% | '
                  f'Test Loss: {test_loss:0.5f}, Test accuracy: {test_accuracy:.2f}%')


decision_boundary(model, X_train, y_train, X_test, y_test)
