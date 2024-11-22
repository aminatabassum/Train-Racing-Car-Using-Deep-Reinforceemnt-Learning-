import numpy as np
import tqdm
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

from ..nn.reinforce_mlp import MLPPolicy
from ..nn.reinforce_cnn import CNNPolicy


class REINFORCEAgent(object):
    def __init__(self, params, model_type="numerical"):
        # create the policy network
        self.policy_net = MLPPolicy(params=params) if model_type == "numerical" else CNNPolicy(params=params)

    def get_action(self, state):
        """ Function to derive an action given a state
            Args:
                state (list): [x/10, y/10, 1]
                
            Returns:
                action index (int), log_prob (ln(\pi(action|state)))
        """
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        probs = self.policy_net(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob


def rolling_average(data, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

class REINFORCEAgentTrainer(object):
    def __init__(self, agent, env, params):
        # Agent object
        self.agent = agent
        
        # Environment object
        self.env = env
        
        # Training parameters
        self.params = params

        # Lists to store the log probabilities and rewards for one episode
        self.saved_log_probs = []
        self.saved_rewards = []

        # Gamma
        self.gamma = params['gamma']
        
        # Small value for returns normalization
        self.eps = params['epsilon']

        # Create the optimizer
        self.optimizer = torch.optim.Adam(self.agent.policy_net.parameters(), lr=params['learning_rate'])

    def update_agent_policy_network(self):
        # We define a list to store the policy loss for each time step
        policy_loss = []
        
        # We define a special list to store the return for each time step
        returns = deque()

        # compute returns for every time step
        G = 0
        for i in range(len(self.saved_rewards) - 1, -1, -1):
            G = self.gamma * G + self.saved_rewards[i]
            returns.appendleft(G)

        # normalize the returns: for stablize the training
        returns = torch.tensor(returns)
        norm_returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for log_prob, r in zip(self.saved_log_probs, norm_returns):
            # compute the loss for each time step
            policy_loss.append(-log_prob * r)

        # We sum all the policy loss
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        
        # after the backpropagation, clear the data to get ready to store the new episode data.
        del self.saved_log_probs[:]
        del self.saved_rewards[:]

        return returns[0].item(), policy_loss.item()

    def rollout(self):
        """ Function to collect one episode from the environment
        """
        # reset the env
        state, done = self.env.reset()
        
        # start rollout
        while True:
            # render the action and log probability given a state (use the feature of the state)
            action, log_prob = self.agent.get_action(state)
            
            # save the log probability to "self.saved_log_probs"
            self.saved_log_probs.append(log_prob)

            # render the next state, reward
            state, reward, done, _, _ = self.env.step(action)
            
            # save the reward to "self.saved_rewards"
            self.saved_rewards.append(reward)

            # check termination
            if done:
                break

    def train(self):
        # number of the training epsiode
        episode_num = self.params['episode_num']
        
        # list to store the returns and losses during the training
        train_returns = []
        train_losses = []
        
        # start the training
        ep_bar = tqdm.trange(episode_num)
        for ep in ep_bar:
            # collect one episode
            self.rollout()

            # update the policy using the collected episode
            G, loss = self.update_agent_policy_network()
            
            # save the returns and losses
            train_returns.append(G)
            train_losses.append(loss)
            
            # add description
            ep_bar.set_description(f"Episode={ep} | Discounted returns = {G} | loss = {loss:.2f}")
            
        # we have to smooth the returns for plotting
        smoothed_returns = rolling_average(np.array(train_returns), window_size=100)
        smoothed_losses = rolling_average(np.array(train_losses), window_size=100)
        
        return smoothed_returns, smoothed_losses