import torch
import numpy as np 
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58],
                   [102, 43, 37], [69, 96, 70 ], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70 ], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70 ]], 
                   dtype = "float32")

targets = np.array([[56, 70], [81, 101],  [119, 133],
                    [22, 37], [103, 119], [56, 70],
                    [81, 101],[119, 133], [22, 37],
                    [103, 119], [56, 70], [81, 101],
                    [119, 133], [22, 37], [103, 119]], 
                    dtype = "float32")

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

"""Define dataset"""
train_ds = TensorDataset(inputs, targets)

"""Define data loader"""
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

"""for xb, yb in train_dl:
    print("batch:")
    print(xb)
    print(yb)"""

"""Define model"""
model = nn.Linear(3, 2)
#print(model.weight)
#print(model.bias)

"""Parameters"""
#list(model.parameters())

"""Define loss function"""
loss_fn = F.mse_loss
#loss = loss_fn(preds, targets)
#print(loss)

"""Define optimizer"""
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

"""Defining the utility function to train the model"""
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):     ## Repeat for given numver of epochs
        for xb,yb in train_dl:          ## Train with batches of data
            preds = model(xb)           ## Generate predictions
            loss = loss_fn(preds, yb)   ## Calculate the loss
            loss.backward()             ## Compute gradients
            opt.step()                  ## Update parameters using gradients
            opt.zero_grad()             ## Reset the gradients to zero

        if (epoch+1) % 10 == 0:   ## Print the progress

            print("Epoch [{}/{}], Loss: {:.4f}". format(epoch+1, num_epochs, loss.item()))

"""Train the model for 100 epochs"""
fit(1000, model, loss_fn, opt)

"""Generate predictions"""
preds = model(inputs)
print(preds)
print(targets)




