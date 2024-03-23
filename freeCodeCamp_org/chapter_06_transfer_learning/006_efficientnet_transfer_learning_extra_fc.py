import importlib
import sys

import torch
from torch import nn, optim
import torchvision.models as models
from torchinfo import summary
from torchmetrics import Accuracy

sys.path.append('../chapter_05_going_modular')

dataset = importlib.import_module('002_data_loaders', 'dataset')
engine = importlib.import_module('003_engine', 'engine')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.efficientnet_b0(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.classifier.append(nn.Linear(in_features=1000, out_features=dataset.number_classes, bias=True))
model.classifier[1].requires_grad_(True)

number_classes = 101
number_epochs = 151
summary(model, input_size=(dataset.batch_size, 3, 224, 224))

# optimizer = optim.Adagrad(params=model.parameters(), lr=0.001)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
# optimizer = optim.SGD(params=model.parameters(), lr=0.0005)
accuracy = Accuracy(num_classes=number_classes, task='multiclass')
loss_function = nn.CrossEntropyLoss().to(device)

engine.train(model=model,
             model_filename='efficientnet_extra_fc.pth',
             use_check_point=True,
             save_best_model=True,
             train_dataloader=dataset.data_loader_full_train,
             test_dataloader=dataset.data_loader_full_test,
             optimizer=optimizer,
             loss_fn=loss_function,
             number_epochs=number_epochs,
             device=device
             )
