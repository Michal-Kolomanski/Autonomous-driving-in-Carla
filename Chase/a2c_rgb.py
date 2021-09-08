"""
The A2C's high-level flow:
1) Initialize the actor's and critic's networks.
2) Use the current policy of the actor to gather n-step experiences from the environment and calculate the n-step return.
3) Calculate the actor's and critic's losses.
4) Perform the stochastic gradient descent optimization step to update the actor and critic parameters.
5) Repeat from step 2.
"""

import numpy as np
import os
from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import settings
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

from carla_env_chase import CarlaEnv

# For RGB camera
from nets.a2c import Actor as DeepActor  # Continuous
from nets.a2c import DiscreteActor as DeepDiscreteActor  # Separate actor
from nets.a2c import Critic as DeepCritic  # Separate critic

from ACTIONS import ACTIONS as ac
from utils import ColoredPrint

# GPU
device = torch.device(settings.SHOULD_USE_CUDA)
seed = 52
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Global settings
port = settings.PORT
action_type = settings.ACTION_TYPE
camera_type = settings.CAMERA_TYPE
load_model = settings.LOAD_MODEL
gamma = settings.GAMMA
lr = settings.LR
use_entropy = settings.USE_ENTROPY
scenario = settings.SCENARIO
model_description = settings.TB_DESCRIPTION

# Tensorboard
writer = SummaryWriter(comment=model_description)

# Transition - the representation of a single transition
"""
:param s: the state
:param value_s: the critic's prediction of the value of state s
:param a: the action taken
:param log_prob_a: the logarithm of the probability of taking action a according to the actor's current policy
"""
Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])


class DeepActorCriticAgent(mp.Process):
    def __init__(self):
        """
        An Advantage Actor-Critic (A2C) agent that uses a Deep Neural Network to represent it's Policy and
        the Value function
        """
        super(DeepActorCriticAgent, self).__init__()
        # Create Carla env
        self.action_type = action_type
        self.camera_type = camera_type
        self.gamma = gamma
        self.lr = lr
        self.use_entropy = use_entropy

        self.env = CarlaEnv(scenario=scenario,  port=port, action_space=self.action_type, camera=self.camera_type,
                            res_x=80, res_y=80, manual_control=False)

        self.environment = self.env  # Carla env
        self.trajectory = []  # Contains the trajectory of the agent as a sequence of transitions
        self.rewards = []  # Contains the rewards obtained from the env at every step
        self.policy = self.discrete_policy  # discrete or continuous

        self.best_mean_reward = - float("inf")  # Agent's personal best mean episode reward
        self.best_reward = - float("inf")
        self.chase_time = []
        self.global_step_num = 0
        self.log = ColoredPrint()
        # For continuous policy
        self.mu = 0
        self.sigma = 0
        # For discrete policy
        self.logits = 0

        self.value = 0
        self.action_distribution = None

        state_shape = [80, 80, 3]
        critic_shape = 1

        if self.action_type == 'discrete':
            self.action_shape = len(ac.ACTIONS_NAMES)
            self.policy = self.discrete_policy
            self.actor = DeepDiscreteActor(state_shape, self.action_shape, device).to(device)
        elif self.action_type == 'continuous':
            self.action_shape = 2
            self.policy = self.multi_variate_gaussian_policy
            self.actor = DeepActor(state_shape, self.action_shape, device).to(device)
        else:
            self.log.err(f"Wrong action type: {self.action_type}, choose discrete or continuous")

        self.critic = DeepCritic(state_shape, critic_shape, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def multi_variate_gaussian_policy(self, obs):
        """
        Calculates a multi-variate gaussian distribution over actions given observations
        :param obs: Agent's observation
        :return: policy, a distribution over actions for the given observation
        """
        mu, sigma = self.actor(obs)
        value = self.critic(obs)
        # Clamp each dim of mu based on the (low,high) limits of that action dim
        [mu[:, i].clamp_(-1, 1) for i in range(self.action_shape)]
        sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7  # Let sigma be (smoothly) + ve
        self.mu = mu.to(torch.device("cuda"))
        self.sigma = sigma.to(torch.device("cuda"))
        self.value = value.to(torch.device("cuda"))
        if len(self.mu.shape) == 0:  # See if mu is a scalar
            self.mu.unsqueeze_(0)

        self.action_distribution = MultivariateNormal(self.mu, torch.eye(self.action_shape).to(device) * self.sigma,
                                                      validate_args=True)
        return self.action_distribution

    def process_action(self, action):
        if self.action_type == 'continuous':
            # Limit the action to lie between the (low, high) limits of the env
            [action[:, i].clamp_(-1, 1) for i in range(self.action_shape)]
        action = action.to(torch.device("cuda"))
        return action.cpu().numpy().squeeze(0)  # Convert to numpy ndarray, squeeze and remove the batch dimension

    def get_action(self, obs):
        action_distribution = self.policy(obs)  # Call to self.policy(obs) also populates self.value with V(obs)
        action = action_distribution.sample()

        log_prob_a = action_distribution.log_prob(action)
        action = self.process_action(action)
        # Store the n-step trajectory while training. Skip storing the trajectories in test mode
        self.trajectory.append(Transition(obs, self.value, action, log_prob_a))  # Construct the trajectory
        return action

    def discrete_policy(self, obs):
        """
        Calculates a discrete/categorical distribution over actions given observations
        :param obs: Agent's observation
        :return: policy, a distribution over actions for the given observation
        """
        logits = self.actor(obs)
        value = self.critic(obs)
        self.logits = logits.to(torch.device("cuda"))
        self.value = value.to(torch.device("cuda"))
        """
        The logits argument will be interpreted as unnormalized log probabilities and can therefore be any real number. 
        It will likewise be normalized so that the resulting probabilities sum to 1 along the last dimension. 
        attr:logits will return this normalized value.
        """
        self.action_distribution = Categorical(logits=self.logits)
        return self.action_distribution

    def calculate_n_step_return(self, n_step_rewards, final_state, done, gamma):
        """
        Calculates the n-step return for each state in the input-trajectory/n_step_transitions
        :param n_step_rewards: List of rewards for each step
        :param final_state: Final state in this n_step_transition/trajectory
        :param done: True rf the final state is a terminal state if not, False
        :return: The n-step return for each state in the n_step_transitions
        """
        g_t_n_s = []
        with torch.no_grad():
            g_t_n = torch.tensor([[0]]).float().to(device) if done else self.critic(final_state)
            for r_t in n_step_rewards[::-1]:  # Reverse order; From r_tpn to r_t
                g_t_n = torch.tensor(r_t).float() + gamma * g_t_n
                g_t_n_s.insert(0, g_t_n)  # n-step returns inserted to the left to maintain correct index order
            return g_t_n_s

    def calculate_loss(self, trajectory, td_targets):
        """
        Calculates the critic and actor losses using the td_targets and self.trajectory
        :param td_targets:
        :param trajectory:
        :return:
        """
        n_step_trajectory = Transition(*zip(*trajectory))
        v_s_batch = n_step_trajectory.value_s  # Critic prediction of the value of state s
        log_prob_a_batch = n_step_trajectory.log_prob_a
        actor_losses, critic_losses, advantages = [], [], []
        for td_target, critic_prediction, log_p_a in zip(td_targets, v_s_batch, log_prob_a_batch):
            writer.add_scalar("Value", critic_prediction, self.global_step_num)
            td_err = td_target - critic_prediction  # td_err is an unbiased estimated of Advantage
            advantages.append(td_err)
            result = - log_p_a * td_err

            actor_losses.append(result)
            critic_losses.append(F.smooth_l1_loss(critic_prediction, td_target))

        if self.use_entropy:
            actor_loss = torch.stack(actor_losses).mean() - self.action_distribution.entropy().mean()
        else:
            actor_loss = torch.stack(actor_losses).mean()

        critic_loss = torch.stack(critic_losses).mean()
        advantage = torch.stack(advantages).mean()

        writer.add_scalar("actor_loss", actor_loss, self.global_step_num)
        writer.add_scalar("critic_loss", critic_loss, self.global_step_num)
        writer.add_scalar("Advantage", advantage, self.global_step_num)
        # writer.add_scalar("log_prob_actions_batch_mean", sum(log_prob_a_batch)/len(log_prob_a_batch), self.global_step_num)
        # writer.add_scalar("v_s_batch", sum(v_s_batch)/len(v_s_batch), self.global_step_num)
        # writer.add_scalar("td_targets", sum(td_targets)/len(td_targets), self.global_step_num)
        # writer.add_scalar("Entropy", self.action_distribution.entropy(), self.global_step_num)
        # writer.add_scalar("Entropy_mean", self.action_distribution.entropy().mean(), self.global_step_num)

        return actor_loss, critic_loss

    def optimize(self, final_state_rgb, done):
        td_targets = self.calculate_n_step_return(self.rewards, final_state_rgb, done, self.gamma)
        actor_loss, critic_loss = self.calculate_loss(self.trajectory, td_targets)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.trajectory.clear()
        self.rewards.clear()

    def train(self):
        # Loading the model
        if load_model:
            self.load(load_model)

        episode_rewards = []  # Every episode's reward
        effective_chase_times = []
        prev_checkpoint_mean_ep_rew = self.best_mean_reward
        num_improved_episodes_before_checkpoint = 0  # To keep track of the num of ep with higher perf to save model

        for episode in range(1000000):
            # Get a single frame form the environment - from a spawn point
            state_rgb = self.environment.reset()
            state_rgb = state_rgb / 255.0  # resize the tensor to [0, 1]

            done = False
            ep_reward = 0.0
            step_num = 0  # used to yield the optimize method
            actions_counter = dict()

            # Calculate how many each action was taken
            for action in ac.ACTIONS_NAMES.values():
                actions_counter[action] = 0

            while not done:
                action = self.get_action(state_rgb)
                if self.action_type == 'discrete':
                    actions_counter[ac.ACTIONS_NAMES[self.environment.action_space[action]]] += 1

                new_state, reward, done = self.environment.step(action)
                new_state = new_state / 255  # resize the tensor to [0, 1]

                self.rewards.append(reward)
                ep_reward += reward
                step_num += 1

                if step_num >= 5 or done:
                    self.optimize(new_state, done)
                    step_num = 0

                state_rgb = new_state
                self.global_step_num += 1
                writer.add_scalar("reward", reward, self.global_step_num)

            if self.action_type == 'discrete':
                print(str(actions_counter))

            episode_rewards.append(ep_reward)
            # The best reward
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
            if np.mean(episode_rewards) > prev_checkpoint_mean_ep_rew:
                num_improved_episodes_before_checkpoint += 1
            if num_improved_episodes_before_checkpoint >= 3:
                prev_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                self.best_mean_reward = np.mean(episode_rewards)
                if not os.path.exists('improved_models'):
                    os.mkdir('improved_models')
                save_path = os.getcwd() + '\improved_models'
                file_name = f"{episode}_" + model_description
                cp_name = os.path.join(save_path, file_name)
                self.save(cp_name)  # Save the model when it improves
                num_improved_episodes_before_checkpoint = 0

            if episode % 100 == 0:  # Save the model per 100 episodes
                self.save(f"100_" + model_description)
            if episode % 250 == 0:
                if not os.path.exists('models'):
                    os.mkdir('models')
                save_path = os.getcwd() + '\models'
                file_name = f"{episode}_" + model_description
                cp_name = os.path.join(save_path, file_name)
                self.save(cp_name)

            print("Episode: {} \t ep_reward:{} \t mean_ep_rew:{}\t best_ep_reward:{}".format(episode,
                                                                                             ep_reward,
                                                                                             np.mean(episode_rewards),
                                                                                             self.best_reward))
            writer.add_scalar("ep_reward", ep_reward, episode)

            effective_chase_times.append(self.env.effective_chase_per)
            avg_chase_per = sum(effective_chase_times) / len(effective_chase_times)
            writer.add_scalar("effective_chase%", self.env.effective_chase_per, episode)
            writer.add_scalar("effective_chase%_mean", avg_chase_per, episode)

    def save(self, name):
        model_file_name = name + ".pth"
        agent_state = {"actor": self.actor.state_dict(),
                       "actor_optimizer": self.actor_optimizer.state_dict(),
                       "critic": self.critic.state_dict(),
                       "critic_optimizer": self.critic_optimizer.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward}
        torch.save(agent_state, model_file_name)
        print("Agent's state is saved to", model_file_name)

    def load(self, name):
        model_file_name = name
        agent_state = torch.load(model_file_name, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(agent_state["actor"])
        self.critic.load_state_dict(agent_state["critic"])
        self.actor.to(device)
        self.critic.to(device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Loaded Advantage Actor-Critic model state from", model_file_name,
              " which fetched a best mean reward of:", self.best_mean_reward,
              " and an all time best reward of:", self.best_reward)


if __name__ == "__main__":
    agent = DeepActorCriticAgent()
    agent.train()
    writer.close()
