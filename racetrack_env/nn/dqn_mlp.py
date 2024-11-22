import torch.nn as nn


class MLPDQN(nn.Module):
    def __init__(self, input_dim, num_hidden_layer, dim_hidden_layer, output_dim):
        super(MLPDQN, self).__init__()

        """CODE HERE: construct your Deep neural network
        """
         # define the input dimension
        self.input_dim = input_dim

        # define the hidden dimension
        self.hidden_num = num_hidden_layer

        # define the number of the hidden layers
        self.hidden_dim = dim_hidden_layer

        # define the output dimension
        self.output_dim = output_dim
      
        self.dqn=nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim),nn.ReLU(),nn.Linear(self.hidden_dim,self.hidden_dim),nn.ReLU(),
                                                                   nn.Linear(self.hidden_dim,self.hidden_dim),nn.ReLU(),nn.Linear(self.hidden_dim,self.output_dim))
                               

    def forward(self,x):
#         print(x.shape)
        y=self.dqn(x)
        return y
      
