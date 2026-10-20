from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import copy
import threading
import queue
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    def __init__(self, ob_dim, ac_dim):
        super().__init__()
        self.fc1 = nn.Linear(ob_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.mean = nn.Linear(128, ac_dim)
        self.log_std = nn.Linear(128, ac_dim)

    def forward(self, ob):
        x = F.relu(self.fc1(ob))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, ob):
        mean, log_std = self.forward(ob)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob, mean


class Critic(nn.Module):
    def __init__(self, ob_dim, ac_dim):
        super().__init__()
        self.fc1 = nn.Linear(ob_dim + ac_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, ob, ac):
        x = torch.cat([ob, ac], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class SACPolicy:
    def __init__(self, ob_dim, ac_dim, action_scale=15.0, device='cuda'):
        # 环境参数
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.action_scale = action_scale
        self.device = device

        # SAC超参数
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.lr_alpha = 1e-5
        self.auto_alpha = True
        self.target_entropy = -ac_dim

        # 前台网络（用于get_action）
        self.actor_front = Actor(ob_dim, ac_dim).to(device)
        self.actor_front.eval()

        # 后台网络（用于训练）
        self.actor_back = Actor(ob_dim, ac_dim).to(device)
        self.critic1_back = Critic(ob_dim, ac_dim).to(device)
        self.critic2_back = Critic(ob_dim, ac_dim).to(device)
        self.critic1_target = Critic(ob_dim, ac_dim).to(device)
        self.critic2_target = Critic(ob_dim, ac_dim).to(device)

        # 初始化target网络
        self.critic1_target.load_state_dict(self.critic1_back.state_dict())
        self.critic2_target.load_state_dict(self.critic2_back.state_dict())

        # 优化器
        self.actor_opt = torch.optim.Adam(self.actor_back.parameters(), lr=self.lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1_back.parameters(), lr=self.lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2_back.parameters(), lr=self.lr_critic)

        # 自动调整alpha
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)

        # 线程安全锁
        self.front_lock = threading.Lock()

        # 计数器
        self.update_count = 0
        self.flush_count = 0

        # 模型保存
        self.save_dir = Path('./data/sac_models')
        self.save_dir.mkdir(exist_ok=True)
        self.max_saved_models = 10
        self.load_latest_model()

        # 异步TensorBoard
        self.log_queue = queue.Queue()
        self.writer = SummaryWriter(log_dir=f'./data/{datetime.now().strftime("%Y%m%d_%H%M%S")}_sac')
        self.logging_active = True
        self.logger_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.logger_thread.start()

    def get_action(self, ob, deterministic=False):
        """前台线程调用，使用前台网络"""

        with torch.no_grad():
            with self.front_lock:
                if deterministic:
                    mean, _ = self.actor_front(ob)
                    action = torch.tanh(mean)
                else:
                    action, _, _ = self.actor_front.sample(ob)

        action *= self.action_scale
        return action

    def update(self, ob, ac, nex_ob, rew, done):
        """后台线程调用，更新后台网络"""

        rew = rew.unsqueeze(-1)
        done = done.unsqueeze(-1)
        # 归一化动作
        ac = ac / self.action_scale

        # 更新Critic
        with torch.no_grad():
            nex_ac, nex_log_prob, _ = self.actor_back.sample(nex_ob)
            nex_q1 = self.critic1_target(nex_ob, nex_ac)
            nex_q2 = self.critic2_target(nex_ob, nex_ac)
            nex_q = torch.min(nex_q1, nex_q2) - self.alpha * nex_log_prob
            target_q = rew + (1 - done) * self.gamma * nex_q

        curr_q1 = self.critic1_back(ob, ac)
        curr_q2 = self.critic2_back(ob, ac)
        critic1_loss = F.mse_loss(curr_q1, target_q)
        critic2_loss = F.mse_loss(curr_q2, target_q)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # 更新Actor
        new_ac, log_prob, _ = self.actor_back.sample(ob)
        q1 = self.critic1_back(ob, new_ac)
        q2 = self.critic2_back(ob, new_ac)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 更新Alpha
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # 软更新target网络
        self._soft_update(self.critic1_back, self.critic1_target)
        self._soft_update(self.critic2_back, self.critic2_target)

        self.update_count += 1

        # 异步记录日志（每5步记录一次）
        if self.update_count % 5 == 0:
            log_data = {
                'step': self.update_count,
                'loss/critic1': critic1_loss.item(),
                'loss/critic2': critic2_loss.item(),
                'loss/actor': actor_loss.item(),
                'value/q1_mean': curr_q1.mean().item(),
                'value/q2_mean': curr_q2.mean().item(),
                'value/target_q_mean': target_q.mean().item(),
                'policy/log_prob_mean': log_prob.mean().item(),
                'policy/alpha': self.alpha,
            }
            if self.auto_alpha:
                log_data['loss/alpha'] = alpha_loss.item()

            self.log_queue.put(log_data)

    def flush(self):
        """每10次update调用一次，线程安全地更新前台网络"""
        # 深拷贝后台actor网络
        new_actor = copy.deepcopy(self.actor_back)
        new_actor.eval()

        # 原子性地替换前台网络指针
        with self.front_lock:
            old_actor = self.actor_front
            self.actor_front = new_actor

        # 旧网络会被垃圾回收
        del old_actor

        self.flush_count += 1

        # 每5000次flush保存一次模型
        if self.flush_count % 5000 == 0:
            self._save_model()

    def _soft_update(self, source, target):
        """软更新target网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def _save_model(self):
        """保存模型，只保留最新的10个"""
        save_path = self.save_dir / f'model_{self.flush_count}.pt'
        torch.save({
            'actor': self.actor_back.state_dict(),
            'critic1': self.critic1_back.state_dict(),
            'critic2': self.critic2_back.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic1_opt': self.critic1_opt.state_dict(),
            'critic2_opt': self.critic2_opt.state_dict(),
            'update_count': self.update_count,
            'flush_count': self.flush_count,
            'log_alpha': self.log_alpha if self.auto_alpha else None,
        }, save_path)

        # 删除旧模型
        saved_models = sorted(self.save_dir.glob('model_*.pt'))
        if len(saved_models) > self.max_saved_models:
            for old_model in saved_models[:-self.max_saved_models]:
                old_model.unlink()

        print(f'Model saved at flush {self.flush_count}')

    def load_latest_model(self):
        """加载最新模型"""
        saved_models = sorted(self.save_dir.glob('model_*.pt'))
        if not saved_models:
            print('No saved model found, starting from scratch')
            return

        latest_model = saved_models[-1]
        checkpoint = torch.load(latest_model, map_location=self.device)

        self.actor_back.load_state_dict(checkpoint['actor'])
        self.critic1_back.load_state_dict(checkpoint['critic1'])
        self.critic2_back.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt'])
        self.critic1_opt.load_state_dict(checkpoint['critic1_opt'])
        self.critic2_opt.load_state_dict(checkpoint['critic2_opt'])
        self.update_count = checkpoint['update_count']
        self.flush_count = checkpoint['flush_count']

        if self.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha = self.log_alpha.exp().item()

        # 同步前台网络
        self.actor_front.load_state_dict(checkpoint['actor'])

        print(f'Loaded model from {latest_model.name} (flush={self.flush_count}, update={self.update_count})')

    def _logging_worker(self):
        """异步日志记录线程"""
        while self.logging_active:
            try:
                log_data = self.log_queue.get(timeout=1.0)
                step = log_data.pop('step')
                for key, value in log_data.items():
                    self.writer.add_scalar(key, value, step)
                self.writer.flush()
            except queue.Empty:
                continue
            except Exception as e:
                print(f'Logging error: {e}')

    def close(self):
        """关闭资源"""
        self.logging_active = False
        self.logger_thread.join(timeout=2.0)
        self.writer.close()
        print('SACPolicy closed')