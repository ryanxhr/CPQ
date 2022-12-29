import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_unit=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.l2 = nn.Linear(hidden_unit, hidden_unit)
        self.l3 = nn.Linear(hidden_unit, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Double_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_unit=256):
        super(Double_Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.l2 = nn.Linear(hidden_unit, hidden_unit)
        self.l3 = nn.Linear(hidden_unit, 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.l5 = nn.Linear(hidden_unit, hidden_unit)
        self.l6 = nn.Linear(hidden_unit, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_unit=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.l2 = nn.Linear(hidden_unit, hidden_unit)
        self.l3 = nn.Linear(hidden_unit, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_unit=256):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_unit)
        self.e2 = nn.Linear(hidden_unit, hidden_unit)

        self.mean = nn.Linear(hidden_unit, latent_dim)
        self.log_std = nn.Linear(hidden_unit, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_unit)
        self.d2 = nn.Linear(hidden_unit, hidden_unit)
        self.d3 = nn.Linear(hidden_unit, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


class CQL_L(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, lmbda=0.75, threshold=30.0, alpha=1.0):
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.reward_critic = Double_Critic(state_dim, action_dim).to(device)
        self.reward_critic_target = copy.deepcopy(self.reward_critic)
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=1e-3)

        self.cost_critic = Critic(state_dim, action_dim).to(device)
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda

        self.alpha = alpha

        self.threshold = threshold
        self.log_lagrangian_weight = torch.zeros(1, requires_grad=True, device=device)
        self.lagrangian_weight_optimizer = torch.optim.Adam([self.log_lagrangian_weight], lr=1e-3)

        self.total_it = 0

    def select_action(self, state):
        with torch.no_grad():
            # state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(device)
            # action = self.actor(state, self.vae.decode(state))
            # q1 = self.reward_critic.q1(state, action)
            # ind = q1.argmax(0)
            # return action[ind].cpu().data.numpy().flatten()
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer / batch
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        cost = torch.sum(torch.abs(action), axis=1).reshape(-1, 1)

        # Reward Critic Training
        with torch.no_grad():
            target_Qr1, target_Qr2 = self.reward_critic_target(next_state, self.actor(next_state))
            # Soft Clipped Double Q-learning
            target_Qr = torch.min(target_Qr1, target_Qr2)
            target_Qr = reward + not_done * self.discount * target_Qr

        current_Qr1, current_Qr2 = self.reward_critic(state, action)
        td_loss = F.mse_loss(current_Qr1, target_Qr) + F.mse_loss(current_Qr2, target_Qr)

        cql_qr1_ood, cql_qr2_ood = self.reward_critic(state, self.actor(state))
        cql_qr1_diff = cql_qr1_ood - current_Qr1
        cql_qr2_diff = cql_qr2_ood - current_Qr2
        cql_loss = self.alpha * (cql_qr1_diff + cql_qr2_diff).mean()

        reward_critic_loss = td_loss + cql_loss

        self.reward_critic_optimizer.zero_grad()
        reward_critic_loss.backward()
        self.reward_critic_optimizer.step()

        # Cost Critic Training
        with torch.no_grad():
            # Compute value of perturbed actions sampled from the VAE
            target_Qc = self.cost_critic_target(next_state, self.actor(next_state))
            target_Qc = cost + not_done * self.discount * target_Qc

        current_Qc = self.cost_critic(state, action)
        cost_critic_loss = F.mse_loss(current_Qc, target_Qc)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        # Pertubation Model / Action Training
        pi_action = self.actor(state)

        # Update through DPG
        self.lagrangian_weight = self.log_lagrangian_weight.exp()
        qr = self.reward_critic.q1(state, pi_action)
        qc = self.cost_critic.q1(state, pi_action)
        actor_loss = (-qr + self.lagrangian_weight * qc).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the Lagrangian weight
        if self.total_it > 150000:
            qc = self.cost_critic.q1(state, pi_action)
            lagrangian_loss = -(self.log_lagrangian_weight * (qc - self.threshold).detach()).mean()

            self.lagrangian_weight_optimizer.zero_grad()
            lagrangian_loss.backward()
            self.lagrangian_weight_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.reward_critic.parameters(), self.reward_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.total_it % 5000 == 0:
            print(f'mean qr value is {qr.mean()}')
            print(f'mean qc value is {qc.mean()}')
            print(f'lagrangian_weight is {self.lagrangian_weight}')