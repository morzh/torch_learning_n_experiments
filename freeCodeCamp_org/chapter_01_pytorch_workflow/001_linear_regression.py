import torch
from torch import nn
import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

print(torch.__version__)

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(len(X), len(y))
train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(y_train), len(y_train), len(y_test), len(y_test))


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param predictions:
    :return:
    """

    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training Data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing Data')
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={'size': 14})
    plt.show()


# plot_predictions()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

with torch.inference_mode():
    """
    faster predictions in inference mode, preferred mode to fo inference
    """
    y_preds = model_0(X_test)

with torch.no_grad():
    """
    similar to inference_model
    """
    y_preds = model_0(X_test)

print(y_preds)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)
loss_values = []
test_loss_values = []
epochs = 1200
for epoch in range(epochs):
    # put model in the training mode
    model_0.train()
    # 1. forward pass
    y_pred = model_0(X_train)
    y_pred_test = model_0(X_test)
    # 2. calculate loss
    loss = loss_fn(y_pred, y_train)
    test_loss = loss_fn(y_pred_test, y_test)
    loss_values.append(float(loss))
    test_loss_values.append(float(test_loss))
    # print(f'Loss: {loss}', end='\r')
    # 3. zero gradients of the optimizer
    optimizer.zero_grad()
    # 4. calculate gradients at the point
    loss.backward()
    # 5.
    optimizer.step()
print('')

# testing
model_0.eval()  # turn off settings not needed for evaluation/testing
print(model_0.state_dict())

plt.plot(range(len(loss_values)), loss_values)
plt.plot(range(len(test_loss_values)), test_loss_values)
plt.show()


with torch.inference_mode():
    y_preds = model_0(X_test)
plot_predictions(predictions=y_preds)

model_0_filename = 'model_0.pth'
torch.save(obj=model_0.state_dict(), f=model_0_filename)

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(model_0_filename))
loaded_model_0.eval()

with torch.inference_mode():
    y_loaded_model_preds = loaded_model_0(X_test)
plot_predictions(predictions=y_loaded_model_preds)
