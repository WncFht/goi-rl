from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import threading
import queue
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class SACPolicy:
    def __init__(self, ob_dim, ac_dim, scale=15.0, device='cuda'):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.scale = scale
        self.device = torch.device(device)

        # Hyperparameters as member variables
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha_lr = 3e-4
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.alpha_init = 0.2
        self.target_entropy = -float(self.ac_dim)

        # Background networks for updates
        self.background_actor = Actor(self.ob_dim, self.ac_dim).to(self.device)
        self.q1 = Critic(self.ob_dim, self.ac_dim).to(self.device)
        self.q2 = Critic(self.ob_dim, self.ac_dim).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)

        # Foreground actor for get_action
        self.foreground_actor = copy.deepcopy(self.background_actor)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.background_actor.parameters(), lr=self.actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.critic_lr)
        self.log_alpha = torch.tensor(np.log(self.alpha_init), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # Counters
        self.update_step = 0
        self.flush_count = 0

        # Asynchronous TensorBoard logging
        self.log_queue = queue.Queue()
        self.writer = SummaryWriter(f'data/{datetime.now().strftime("%Y%m%d_%H%M%S")}_sac')
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

        # Model saving
        self.save_dir = './data/sac_models'
        os.makedirs(self.save_dir, exist_ok=True)
        self.max_saves = 10
        self.saved_files = []
        self._load_latest()

    def _load_latest(self):
        files = sorted([f for f in os.listdir(self.save_dir) if f.startswith('sac_') and f.endswith('.pth')])
        if files:
            latest_path = os.path.join(self.save_dir, files[-1])
            checkpoint = torch.load(latest_path, map_location=self.device)
            self.background_actor.load_state_dict(checkpoint['background_actor'])
            self.foreground_actor.load_state_dict(checkpoint['background_actor'])
            self.q1.load_state_dict(checkpoint['q1'])
            self.q2.load_state_dict(checkpoint['q2'])
            self.target_q1.load_state_dict(checkpoint['target_q1'])
            self.target_q2.load_state_dict(checkpoint['target_q2'])
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
            # Optimizers not loaded; reinitialized

    def _save_model(self):
        path = os.path.join(self.save_dir, f'sac_{self.flush_count}.pth')
        torch.save({
            'background_actor': self.background_actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'log_alpha': self.log_alpha.data
        }, path)
        self.saved_files.append(path)
        if len(self.saved_files) > self.max_saves:
            old_path = self.saved_files.pop(0)
            os.remove(old_path)

    def get_action(self, ob):
        with torch.no_grad():
            mu, log_std = self.foreground_actor(ob)
            std = log_std.exp()
            z = torch.normal(mu, std)  # Sample for stochastic policy
            action = torch.tanh(z) * self.scale
        return action

    def update(self, ob, ac, nex_ob, rew, done):
        self.update_step += 1

        rew = rew.unsqueeze(1)
        done = done.unsqueeze(1)

        # Compute target Q
        with torch.no_grad():
            next_mu, next_log_std = self.background_actor(nex_ob)
            next_std = next_log_std.exp()
            next_z = next_mu + next_std * torch.randn_like(next_mu)
            next_action = torch.tanh(next_z) * self.scale
            next_log_pi = (torch.log(next_std) + 0.5 * np.log(2 * np.pi) + 0.5 * ((next_z - next_mu) / next_std).pow(
                2)).sum(dim=1, keepdim=True) * -1
            next_log_pi -= torch.log((1 - next_action.pow(2) + 1e-6).clamp(0, 1)).sum(dim=1, keepdim=True)
            alpha = self.log_alpha.exp()
            next_q1 = self.target_q1(nex_ob, next_action)
            next_q2 = self.target_q2(nex_ob, next_action)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_pi
            target_q = rew + (1 - done.float()) * self.gamma * next_q

        # Critic updates
        q1 = self.q1(ob, ac)
        q2 = self.q2(ob, ac)
        critic_loss1 = (q1 - target_q).pow(2).mean()
        critic_loss2 = (q2 - target_q).pow(2).mean()

        self.q1_optimizer.zero_grad()
        critic_loss1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        critic_loss2.backward()
        self.q2_optimizer.step()

        # Actor update
        mu, log_std = self.background_actor(ob)
        std = log_std.exp()
        z = mu + std * torch.randn_like(mu)
        action = torch.tanh(z) * self.scale
        log_pi = (torch.log(std) + 0.5 * np.log(2 * np.pi) + 0.5 * ((z - mu) / std).pow(2)).sum(dim=1,
                                                                                                keepdim=True) * -1
        log_pi -= torch.log((1 - action.pow(2) + 1e-6).clamp(0, 1)).sum(dim=1, keepdim=True)

        q1_pi = self.q1(ob, action)
        q2_pi = self.q2(ob, action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = - (self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)
            for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

        # Asynchronous logging every 5 updates
        if self.update_step % 5 == 0:
            self.log_queue.put({
                'Loss/Actor': actor_loss.item(),
                'Loss/Critic1': critic_loss1.item(),
                'Loss/Critic2': critic_loss2.item(),
                'Loss/Alpha': alpha_loss.item(),
                'Alpha': alpha.item(),
                'Entropy/LogPiMean': log_pi.mean().item(),
                'Q/MeanQ1': q1.mean().item(),
                'Q/MeanQ2': q2.mean().item(),
                'Q/TargetMean': target_q.mean().item()
            })

    def flush(self):
        self.flush_count += 1
        # Deep copy background actor and swap pointer atomically
        new_actor = copy.deepcopy(self.background_actor)
        self.foreground_actor = new_actor
        # Save every 5000 flushes
        if self.flush_count % 5000 == 0:
            self._save_model()

    def _log_worker(self):
        while True:
            log_data = self.log_queue.get()
            global_step = self.update_step
            for tag, value in log_data.items():
                self.writer.add_scalar(tag, value, global_step)
            self.log_queue.task_done()


class Actor(nn.Module):
    def __init__(self, ob_dim, ac_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(ob_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.mu_head = nn.Linear(128, ac_dim)
        self.log_std_head = nn.Linear(128, ac_dim)

    def forward(self, ob):
        x = torch.relu(self.fc1(ob))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std


class Critic(nn.Module):
    def __init__(self, ob_dim, ac_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(ob_dim + ac_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, 1)

    def forward(self, ob, ac):
        x = torch.cat([ob, ac], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.q_head(x)