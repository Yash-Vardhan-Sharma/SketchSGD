import torch
import numpy as np
from sketched_optimizer import SketchedSGD, SketchedSum, SketchedModel

from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import torch.nn as nn

torch.manual_seed(42)

# Create Dataset (train and test)

train_dataset = EMNIST(root="data/", split="byclass", download=True, train=True, 
                transform=tt.Compose([
                    tt.ToTensor()
                ]))

test_dataset = EMNIST(root="data/", split="byclass", download=True, train=False, 
                transform=tt.Compose([
                    tt.ToTensor()
                ]))
                
batch_size = 512

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

total_batches = len(train_dataloader)

device = 'cuda'

# Define the test funtion

def evaluate(model):
  model.eval()
  cor , total = 0,0
  for data,labels in test_dataloader:

    data , labels = data.to(device) , labels.to(device)
    
    ypred = model(data.reshape(-1,28*28))
    ypred = torch.argmax(ypred, dim=1)

    cor  += torch.sum(ypred==labels).item()
    total += len(ypred==labels)

  acc = cor*100/total
  return acc



#Instantiate pytorch model

layers = []
layers.append(nn.Linear(28*28, 500))
layers.append(nn.ReLU())
layers.append(nn.Linear(500 , 256))
layers.append(nn.ReLU())
layers.append(nn.Linear(256,62))


model = nn.Sequential(*layers)

#Wrap the model

model = SketchedModel(model)
model = model.to(device)

#Instantiate the optimizer

opt = torch.optim.SGD(model.parameters(), lr=0.001)

#Wrap the optimizer

opt = SketchedSGD(opt, k=133968 ,accumulateError=True, p1=0, p2=4)

summer = SketchedSum(opt, c=5, r=20, numWorkers=4)

print("EMNIST Training")

criterion = nn.CrossEntropyLoss(reduction='none')

# Training for 1 epoch
for i in range(1): 

  for j , (data , label) in enumerate(train_dataloader):
    opt.zero_grad()

    X = data.reshape(-1 , 28*28)
    y = label

    X = X.to(device)
    y = y.to(device)

    yPred = model(X)
    yPred_ = torch.argmax(yPred , dim=1)

    loss = criterion(yPred,y)

    loss = summer(loss)

    print("[{}:{}/{}] Loss: {}".format(i, j , total_batches , loss.item()/batch_size))
    print("Train Accuracy:", 100 * torch.sum(yPred_ == y).item() / len(yPred_==y) )

    loss.backward()

    opt.step()

acc = evaluate(model)

print("Test accuracy:", acc)
