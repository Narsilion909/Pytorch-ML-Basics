import numpy as np 
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F

"""Download training dataset"""
dataset = MNIST(root = "data/", download = True)
#print(len(dataset))

test_dataset = MNIST(root="data/", train = False)
#print(len(test_dataset))

"""Getting the image of first sample"""
#image, label = dataset[0]
#plt.imshow(image, cmap="gray")
#plt.show()
#print("Label: ", label)

"""MNIST dataset (images and labels)"""
dataset = MNIST(root="data/",
                train = True,
                transform = transforms.ToTensor())

img_tensor, label = dataset[0]
#print(img_tensor.shape, label)
#print(img_tensor[:, 10:15, 10:15])
#print(torch.max(img_tensor), torch.min(img_tensor))

"""Defining a function that randomly picks a given fraction 
   of the images for the validation set"""
def split_indices(n, val_pct):
    n_val = int(val_pct*n)   ## Determining size of the validation set
    idxs = np.random.permutation(n)   ## Creating random permutation of 0 to n-1
    return idxs[n_val:], idxs[:n_val]

train_indices, val_indices = split_indices(len(dataset), val_pct = 0.2)
#print(len(train_indices), len(val_indices))
#print("Sample validation indices: ", val_indices[:5])

"""Defining data samplers and data loaders"""
batch_size = 100

train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset,
                          batch_size,
                          sampler=train_sampler)
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset,
                        batch_size,
                        sampler=val_sampler)

"""Creating logistic regression model"""
input_size = 28*28
num_classes = 10
#model = nn.Linear(input_size, num_classes)
#print(model.weight.shape)
#print(model.bias.shape)

"""for images, labels in train_loader:
    #images = images.reshape(-1, 784)
    print(labels)
    print(images.shape)
    outputs = model(images)
    break"""

class MnistModel (nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward (self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

model = MnistModel()
for images, labels in train_loader:
    outputs = model(images)
    break
#print("outputs.shape: ", outputs.shape)
#print("Sample Outputs: \n", outputs[:2].data)

"""Applying the softmax function"""
probs = F.softmax(outputs, dim=1)   ## Applying softmax 
#print("Sample probabilities:\n", probs[:2].data)
#print("Sum: ", torch.sum(probs[0]).item())

max_probs, preds = torch.max(probs, dim=1)
#print(preds)
#print(max_probs)

"""Evaluation Metric and Loss Function"""
#print(labels == preds)
def accuracy (labels, preds):
    return torch.sum(labels == preds).item() / len(labels)
#print(accuracy(labels, preds))

loss_fn = F.cross_entropy   ## cross_entropy has softmax integrated in
loss = loss_fn(outputs, labels)
#print(loss)

"""Optimizer"""
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""Training the model"""
def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None: 
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
                   for xb,yb in valid_dl]   ## Passing each validation batch through the model
        losses, nums, metrics = zip(*results)   ## Separating losses, counts and metrics
        total = np.sum(nums)   ## Total size of the dataset
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

#val_loss, total, val_acc = evaluate(model, loss_fn, val_loader, metric=accuracy)
#print("Loss: {:.4f}, Accuracy: {:.4f}".format(val_loss, val_acc))

def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
    for epoch in range(epochs):
        ## Training
        for xb,yb in train_dl:
            loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)

        ## Evaluation
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        ## Print progress
        if metric is None:
            print("Epoch [{}/{}], Loss: {:.4f}"
                  .format(epoch+1, epochs, val_loss))
        else:
            print("Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}"
                  .format(epoch+1, epochs, val_loss, metric.__name__, val_metric))

"""Training the model for 5 epochs"""              
fit(5, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)

#accuracies = []
#plt.plot(accuracies, "-x")
#plt.xlabel("epoch")
#plt.ylabel("accuracy")
#plt.title("Accuracy vs No. of epochs")

"""Defining test dataset"""
test_dataset = MNIST(root="data/",
                     train=False,
                     transform=transform.ToTensor())

"""Defining a fucntion that makes prediction for a single image"""
def predictImage(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item

test_loader = DataLoader(test_dataset, batch_size=200)
test_loss, total, test_acc= evaluate(model, loss_fn, test_loader, metric=accuracy)
print("Loss: {:.4f}, Accuracy: {:.4f}".format(test_loss, test_acc))

"""Saving weights and biases of the trained model"""
torch.save(model.state_dict(), "mnist-logistic.pth")
print(model.state_dict)

"""Loading the presaved weights and biases to a new model"""
model2 = MnistModel()
model2.load_state_dict(torch.load("mnist-logistic.pth"))
print(model2.state_dict())