
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, output_size):
        super().__init__()

        output_size = 1
        self.fc1 = nn.Linear(input_size, n_hidden1) 
        self.fc2 = nn.Linear(n_hidden1, n_hidden2) 
        self.fc3 = nn.Linear(n_hidden2, output_size) 

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x)) # leakyrelu
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        #x = self.sigmoid(x)
        return x


    
