import torch.nn as nn
import torch.nn.functional as F


def get_input_shape(eye_sight):
    return (eye_sight * 2 + 1) ** 2


class MLPPolicy(nn.Module):
    def __init__(self, params):
        super(MLPPolicy, self).__init__()
        self.affine1 = nn.Linear(get_input_shape(params['eye_sight']), 128)
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.affine2 = nn.Linear(128, 5)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
