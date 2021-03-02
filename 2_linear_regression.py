import numpy as np 
import torch

inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70 ]], dtype = "float32")

targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype = "float32")

"""Convert inputs and targets to tensors"""
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
#print(inputs)
#print(targets)

"""Defining weights and biases"""
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
#print(w)
#print(b)

"""Defining the model"""
def model(x):
    return x @ w.t() + b

"""Generating predictions"""
preds = model(inputs)
print(preds)

"""Compare with targets"""
print(targets)

"""Defining the MSE loss function"""
def mse(t1, t2):
    diff = t1-t2
    return torch.sum(diff*diff) / diff.numel()

loss = mse(preds, targets)
#print("Loss = " + str(loss))

"""Calculating gradients"""
loss.backward()
#print(w)
#print(b)
#print(w.grad)
#print(b.grad)

"""Adjusting weights & biases and reseting the gradients"""
learning_rate = 1e-5
with torch.no_grad():
    w -= w.grad * learning_rate
    b -= b.grad * learning_rate
    w.grad.zero_()
    b.grad.zero_()   
print(w)
print(b)

"""Training model for 100 epochs"""
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
        w.grad.zero_()
        b.grad.zero_() 

print("Loss = " + str(loss))
print(preds)
print(targets)