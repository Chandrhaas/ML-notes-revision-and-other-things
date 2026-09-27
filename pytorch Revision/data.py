""" 
    creating custom data sets, we need three functions init, len and getitem

    in init we intitalize the directory

    in len we just return the length of dataset

    in getitem loads and returns sample value at idx from the dataset

    dataset retrives the sample and labels one at a time we use dataloader to make minibatches to use the multiprocessing of python,
    dataloader makes the data into mini batches and iterable 
 
 
 """
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class customdata(Dataset):
    def __init__(self,xtrain, ytrain,transform=None,target_transform=None): 
        #transform modifies the features and target transform modifies the labels 
        self.x=xtrain
        self.y=ytrain
        self.transform=transform
        self.target_transform=target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X= self.x[index]
        Y=self.y[index]
        X=self.transform(X)
        Y=self.target_transform(Y)
        return X,Y

"""
after making the class we need to make the object, in the object we pass the x, y, ... whatever is input requirements of init
when we need the dataset as a whole we just pass the object we made into dataloader
"""
a = torch.tensor([1,2])
b = torch.tensor([3,4])

dataset_object = customdata(
    a,b,None,None
)


training = DataLoader(dataset_object,batch_size=1,shuffle=True) #same for test data

for a,b in training:
    #training
    c=a+b

