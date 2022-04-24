import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 128

class Memory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.done_mask = []
        self.log_probs = []

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.done_mask.clear()
        self.log_probs.clear()

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()

        self.input_robot_ob_block = nn.Sequential(
            nn.Linear(obs_dim-3, HIDDEN_SIZE//2),
            nn.Tanh()
        )

        self.input_next_anchor_block = nn.Sequential(
            nn.Linear(3, HIDDEN_SIZE//2),
            nn.Tanh()
        )

        self.mu = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, act_dim),
            nn.Tanh()
        )

        self.std = nn.Parameter(torch.ones(act_dim))

    def forward(self, obs):
        out_robot = self.input_robot_ob_block(obs[:, :-3])
        out_next_anchor = self.input_next_anchor_block(obs[:, -3:])
        action_mu = self.mu(torch.cat((out_robot, out_next_anchor), dim=1))
        action_std = self.std

        return action_mu, action_std

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()

        self.input_robot_ob_block = nn.Sequential(
            nn.Linear(obs_dim - 3, HIDDEN_SIZE // 2),
            nn.ReLU()
        )

        self.input_next_anchor_block = nn.Sequential(
            nn.Linear(3, HIDDEN_SIZE // 2),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, obs):
        out_robot = self.input_robot_ob_block(obs[:, :-3])
        out_next_anchor = self.input_next_anchor_block(obs[:, -3:])
        state_value = self.value(torch.cat((out_robot, out_next_anchor), dim=1))

        return state_value

class PPO:
    def __init__(self, obs_dim, act_dim, lmbda=0.98, gamma=0.99, clip_eps=0.2, c1=0.5, c2=0.01, lr=1e-4, K_epochs=10):
        self.lmbda = lmbda
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.lr = lr
        self.K_epochs = K_epochs

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        self.critic_criterion = nn.MSELoss()

    def select_action(self, obs, memory):
        obs_tensor = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        mu, std = self.actor(obs_tensor)
        dist = Normal(mu, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        memory.observations.append(obs_tensor.flatten())
        memory.actions.append(action.flatten())
        memory.log_probs.append(action_log_prob.flatten())

        return action.detach().cpu().numpy().flatten()


    def update(self, memory):
        observations = torch.stack(memory.observations).to(device).detach()
        actions = torch.stack(memory.actions).to(device).detach()
        rewards = torch.FloatTensor(memory.rewards).to(device).detach()
        masks = torch.FloatTensor(memory.done_mask).to(device).detach()
        old_policy = torch.stack(memory.log_probs).to(device).detach()

        old_values = self.critic(observations)
        returns, advantages = self._get_gae(rewards, masks, old_values)
        advantages = advantages.unsqueeze(dim=1)

        # optimize policy for K epochs
        for _ in range(self.K_epochs):
            # calculate actor loss
            new_mu, new_std = self.actor(observations)
            new_dist = Normal(new_mu, new_std)
            new_policy = new_dist.log_prob(actions)

            ratio = torch.exp(new_policy - old_policy)
            surr_loss1 = ratio * advantages.detach()
            surr_loss2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages.detach()
            actor_loss = -torch.min(surr_loss1, surr_loss2)

            # calculate critic loss
            new_values = self.critic(observations)
            critic_loss = self.c1 * self.critic_criterion(new_values, returns)

            # calculate entropy bonus
            entropy_bonus = -self.c2 * new_dist.entropy()

            # update actor & critic
            loss = actor_loss + critic_loss + entropy_bonus

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        print(f'ppo loss: {loss.mean().item()}')

    def _get_gae(self, rewards, masks, values):
        returns = torch.zeros_like(rewards).to(device)
        advantages = torch.zeros_like(rewards).to(device)

        cumulated_returns = 0.
        cumulated_advantages = 0.
        next_value = 0.

        for t in reversed(range(len(rewards))):
            cumulated_returns = rewards[t] + self.gamma * cumulated_returns * masks[t]
            returns[t] = cumulated_returns

            td_error = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            cumulated_advantages = td_error + self.gamma * self.lmbda * cumulated_advantages * masks[t]
            advantages[t] = cumulated_advantages

            next_value = values[t]

        advantages = (advantages - advantages.mean()) / advantages.std()

        return returns, advantages

