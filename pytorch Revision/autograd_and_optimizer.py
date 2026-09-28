"""
autograd-
as we have seen in the lifecycle of the nerual network (refer notes.txt)
model does the forward pass, then prediction and finally loss is calculated, to make the weights better we need to know
which direction to shift them for that we need to find the gradients, autograd help us find those gradients (it does not update the weights
only finds the gradients, updating is optimizer territory)

optimizer-
optimizer moves the weight according to the direction of gradients and magnitude of the learning rate

learning rate, batch size and epochs are hyperparameters that we need to tune
"""


#assume a neural network, define an optimizer,loss fn
import torch

def loss_fn(x,y):
    return 0

optimizer = torch.optim.Adam(model.parameter(),lr=0.1) 

data = [] # assume this is the dataset
epochs =10;

for i in epochs:
    model.train() #puts the model into training mode good for where dataset in training has a specific nuance eg- dropout 
    for x,y in data:
        optimizer.zero_grad() #this step makes the previous grads=0 so they do not accumulate
        prediction = model(x)
        loss = loss_fn(prediction,y)
        loss.backwards() #this step is the backprop or gradient calculation step
        optimizer.step() #optimizer moves the weights
