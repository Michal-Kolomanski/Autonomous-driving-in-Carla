import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


# NN IMPLEMENTATION (CNN for rgb image 80x80x3 and LINEAR for metrics tensor 1x2) then concat with torch.cat function

class Actor(nn.Module):
    def __init__(self, actor_shape=2, device=torch.device("cuda")):
        super(Actor, self).__init__()
        self.device = device
        self.convolution1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=0),
                                          nn.ReLU())
        self.convolution2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.ReLU())
        self.convolution3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.fully_connected_convolutional = nn.Linear(in_features=64 * 7 * 7, out_features=256)

        self.fully_connected_metrics = nn.Linear(in_features=2, out_features=128)
        self.fully_connected_metrics2 = nn.Linear(in_features=128, out_features=256)

        self.mean_linear = nn.Linear(512, actor_shape)
        self.sigma_linear = nn.Linear(512, actor_shape)

    def forward(self, state_rgb, metrics):
        state_rgb.requires_grad_()
        metrics.requires_grad_()
        state_rgb = state_rgb.to(self.device)
        metrics = metrics.to(self.device)
        state_rgb = self.convolution1(state_rgb)
        state_rgb = self.convolution2(state_rgb)
        state_rgb = self.convolution3(state_rgb)
        state_rgb = state_rgb.view(state_rgb.shape[0], -1)
        state_rgb = self.fully_connected_convolutional(state_rgb)  # 256 rgb out features

        metrics = F.relu(self.fully_connected_metrics(metrics))
        metrics = F.relu(self.fully_connected_metrics2(metrics))  # 256 metrics out features

        input_ = torch.cat((state_rgb, metrics), 1)  # 512 input features

        # mean = torch.tanh(self.mean_linear(input_))
        mean = self.mean_linear(input_)
        sigma = self.sigma_linear(input_)
        critic_value = self.critic_linear(input_)

        return critic_value, sigma, mean


class Action(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super(Action, self).__init__()
        self.device = device

    def forward(self, mean, sigma):
        mean = torch.tanh(mean)
        sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7
        mean = mean.to(self.device)
        sigma = sigma.to(self.device)
        action_distribution = MultivariateNormal(mean, torch.eye(2) * sigma, validate_args=True)
        action = action_distribution.sample()
        log_prob_a = action_distribution.log_prob(action)
        action = torch.tanh(action)
        action = action.numpy().squeeze(0)
        action = action.to(self.device)
        return action


class AI(nn.Module):
    def __init__(self, actor_critic, action):
        super(AI, self).__init__()
        self.actor_critic = actor_critic
        self.action = action

    def forward(self, state_rgb, metrics):
        critic_value, sigma, mean = self.actor_critic(state_rgb, metrics)
        critic_value.to(self.action.device)
        action = self.action(mean, sigma)
        return action, critic_value
