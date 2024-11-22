import torch.nn as nn
import torch.nn.functional as F


def get_flattened_shape(eye_sight):
    two_x_plus_one = eye_sight * 2 + 1
    return 5 * (two_x_plus_one - 2) * (two_x_plus_one - 2)


class CNNPolicy(nn.Module):
    def __init__(self, params):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(5)
        self.flatten = nn.Flatten()
        self.affine2 = nn.Linear(get_flattened_shape(params['eye_sight']), 5)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.leaky_relu(self.batchnorm1(self.conv1(x)))
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)))
        x = self.flatten(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
