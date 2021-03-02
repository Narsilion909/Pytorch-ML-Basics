import torch
import numpy as np
 
"""Creating tensors"""

t1 = torch.tensor(4.)  ## Floating number
t2 = torch.tensor([1., 2, 3, 4])   ## Vector
t3 = torch.tensor([[5., 6], [7, 8], [9, 10]])  ## Matrix
t4 = torch.tensor([
                  [[11, 12, 13],
                   [13, 14, 15]],
                  [[15, 16,17],
                   [17, 18, 19.]]])   ## 3 dimentional array

print([t1.shape], [t2.shape], [t3.shape], [t4.shape])
print([t1.dtype], [t2.dtype], [t3.dtype], [t4.dtype])

""" Arithmetic operations"""

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True) ## We can compute derivative when we set requires_grad = True
b = torch.tensor(5., requires_grad=True)

y = w * x + b

"""Computing derivatives"""

## We can atomatically compte the derivative of y :
y.backward()  ## The derivatives of 'y' w.r.t are stored in the '.grad property of the respective tensors.

"""Displaying gradients"""

print("dy/dx = ", x.grad)  ## Since we didn't set requires_grad to True for x, dy/dx will be None.
print("dy/dx = ", w.grad)
print("dy/dx = ", b.grad)

"""Converting from-to numpy array-tensor"""

a = np.array([[1,2], [3,4.]])
b = torch.from_numpy(a)
#b = torch.tensor(a)  ## Creates a copy of the data
c = b.numpy()

print(a.dtype, b.dtype, c.dtype)

