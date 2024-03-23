from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import colormaps
import torch, torchvision
from torchmetrics import Accuracy
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

number_clusters = 4
number_features = 2
cluster_centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
random_seed = 42
number_samples = 30000
clusters_std = 1.5
plot_dataset = True


X, y = make_blobs(n_samples=number_samples, n_features=number_features,  centers=number_clusters,
                  cluster_std=clusters_std, random_state=random_seed)


X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

if plot_dataset:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Train")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap=colormaps['RdYlBu'])
    # plt.title("Estimated number of clusters: %d" % number_clusters)

    plt.subplot(1, 2, 2)
    plt.title("Test")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=1, cmap=colormaps['RdYlBu'])

    plt.tight_layout()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch version:', torch.__version__, 'device:', device)


class BlobsModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layers_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            # nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layers_stack(x)


blob_model = BlobsModel(input_features=number_features,
                        output_features=number_clusters,
                        hidden_units=10).to(device)


loss_function = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=model.parameters(), lr=0.25)
optimizer = torch.optim.Adam(params=blob_model.parameters(), lr=0.08)
# print(model_0.state_dict())


# def accuracy(y_true, y_pred):
#     matches = torch.eq(y_true, y_pred).sum().item()
#     accuracy_percents = matches / len(y_pred) * 100
#     return accuracy_percents

accuracy = Accuracy(task="multiclass", num_classes=number_clusters).to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

number_epochs = 700
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


for epoch in range(number_epochs):
    blob_model.train()

    y_logits = blob_model(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_function(y_logits, y_train)
    acc = accuracy(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    blob_model.eval()
    with torch.inference_mode():
        test_logits = blob_model(X_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_function(test_logits, y_test)
        test_accuracy = 100 * accuracy(test_preds, y_test)

        if epoch % 100 == 0:
            print(f'Epoch: {epoch} | Loss {test_loss:.5f}, Accuracy: {acc:.2f}% | '
                  f'Test Loss: {test_loss:0.5f}, Test accuracy: {test_accuracy:.2f}%')


decision_boundary(blob_model, X_train, y_train, X_test, y_test)
