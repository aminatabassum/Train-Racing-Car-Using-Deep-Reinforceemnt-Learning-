import torch.nn as nn
import torch.nn.functional as F

class CNNDQN(nn.Module):
    def __init__(self):
        super(CNNDQN, self).__init__()

        """CODE HERE: construct your Deep neural network
        """
        
        self.conv1 = nn.Conv2d(1, 3, 3, 1,1)
        self.conv2 = nn.Conv2d(3, 5, 3)
        self.flatten = nn.Flatten()
        length=2 * 2 + 1
        resha_flatten=5 * (length - 2) * (length - 2)
        self.conv3 = nn.Linear(resha_flatten, 5)
        

    def forward(self, x):
        print(x.shape)
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x = self.flatten(x)
        action_scores = self.conv3(x)
        return F.softmax(action_scores, dim=1)
      
    
  
      
