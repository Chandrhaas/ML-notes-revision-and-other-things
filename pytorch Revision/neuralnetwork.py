"""
torch.nn module has everything we need to buil our own nn
we build the nn in __init__() fn
"""

import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        """
        can use sequential directly here or first stack layers in an array and then use sequential on them
        abc = Sequential(
        nn.linear(dim,dim)
        nn.relu()
        ...
        ..
        )

        """

    def forward(self,x):
        ans = self.abc(x)
        return ans

"""
to run the neural net we can make an object of the class, eg- model= neuralneat(....)
after that we just need to pass the input in this case model(x)
"""