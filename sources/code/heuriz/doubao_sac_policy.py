import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import threading
import copy
from typing import Optional, Tuple, Dict, Any
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import queue
import time
import collections


class LoggerThread(threading.Thread):
    """优化的日志记录线程（批量写入+缓存）"""

    def __init__(self, log_dir: str, max_queue_size: int = 10000, flush_interval: int = 30):
        super().__init__(daemon=True)
        self.writer = SummaryWriter(log_dir, flush_secs=30)  # 减少flush频率
        self.log_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self.flush_interval = flush_interval

        # 批量缓存
        self.batch_cache = collections.defaultdict(list)
        self.cache_size = 0
        self.max_cache_size = 100  # 缓存100条再写入
        self.last_flush_time = time.time()

        self.start()

    def log(self, tag: str, value: float, step: int):
        """非阻塞的日志记录接口"""
        try:
            self.log_queue.put_nowait((tag, value, step))
        except queue.Full:
            pass  # 直接丢弃，避免阻塞

    def _process_batch(self):
        """批量写入日志"""
        if not self.batch_cache:
            return

        try:
            for tag, entries in self.batch_cache.items():
                # 按step排序
                entries.sort(key=lambda x: x[1])
                steps = [e[1] for e in entries]
                values = [e[0] for e in entries]

                # 使用add_scalars批量添加（比多次add_scalar更高效）
                self.writer.add_scalar(tag, values[-1], steps[-1])  # 只记录最新值

            self.batch_cache.clear()
            self.cache_size = 0
        except Exception as e:
            print(f"Logger batch error: {e}")

    def run(self):
        """优化的线程主循环"""
        while not self._stop_event.is_set():
            try:
                # 批量处理队列中的日志
                batch_count = 0
                while batch_count < 50 and not self.log_queue.empty():
                    tag, value, step = self.log_queue.get(timeout=0.01)
                    self.batch_cache[tag].append((value, step))
                    self.cache_size += 1
                    batch_count += 1

                # 缓存达到阈值或超时则写入
                current_time = time.time()
                if (self.cache_size >= self.max_cache_size or
                        current_time - self.last_flush_time > self.flush_interval):
                    self._process_batch()
                    self.last_flush_time = current_time

                # 短暂休眠，减少CPU占用
                time.sleep(0.001)

            except queue.Empty:
                time.sleep(0.01)  # 队列为空时休眠更长时间
            except Exception as e:
                print(f"Logger error: {e}")
                time.sleep(0.1)

        # 最后清理缓存
        self._process_batch()
        self.writer.close()

    def stop(self):
        """停止日志线程"""
        self._stop_event.set()
        self.join(timeout=5)


class DoubaoSACPolicy(nn.Module):
    def __init__(self,
                 ob_dim: int,
                 ac_dim: int,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.05,  #大动作的时候，探索已经很大了。。。
                 target_update_freq: int = 10,
                 log_dir: Optional[str] = None,
                 log_frequency: int = 10):  # 每10步记录一次日志
        """
        SAC Policy实现（双网络架构，线程安全）
        :param ob_dim: 观测维度
        :param ac_dim: 动作维度
        :param hidden_dim: 隐藏层维度
        :param lr: 学习率
        :param gamma: 折扣因子
        :param tau: 软更新系数（用于旧网络同步）
        :param alpha: 熵正则化系数
        :param target_update_freq: flush前的更新次数阈值
        :param log_dir: 日志目录（None则自动生成）
        :param log_frequency: 日志记录频率（步）
        """
        super().__init__()
        # 超参数配置
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_freq = target_update_freq
        self.log_frequency = log_frequency

        # 日志系统初始化
        if log_dir is None:
            log_dir = f'data/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            time.sleep(1) #保证这个logger和另外一个不要冲突命名
        self.logger = LoggerThread(log_dir)
        self.global_step = 0

        # 网络初始化：前台网络（用于get_action，只读）、后台网络（用于update）
        self.actor_front: Optional[Actor] = None
        self.critic1_front: Optional[Critic] = None
        self.critic2_front: Optional[Critic] = None

        self.actor_back: Optional[Actor] = None
        self.critic1_back: Optional[Critic] = None
        self.critic2_back: Optional[Critic] = None

        self._init_networks()

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor_back.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1_back.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2_back.parameters(), lr=lr)

        # 线程安全相关
        self._lock = threading.Lock()  # 仅用于后台网络更新时的保护
        self._update_count = 0  # 后台更新计数器
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将所有网络移到设备上
        self.to(self._device)

    def _init_networks(self):
        """初始化前后台网络"""
        # 后台网络（训练用）
        self.actor_back = Actor(self.ob_dim, self.ac_dim, self.hidden_dim)
        self.critic1_back = Critic(self.ob_dim, self.ac_dim, self.hidden_dim)
        self.critic2_back = Critic(self.ob_dim, self.ac_dim, self.hidden_dim)

        # 前台网络（推理用）- 初始时复制后台网络参数
        self.actor_front = copy.deepcopy(self.actor_back)
        self.critic1_front = copy.deepcopy(self.critic1_back)
        self.critic2_front = copy.deepcopy(self.critic2_back)

        # 前台网络设为eval模式，禁止梯度计算
        self.actor_front.eval()
        self.critic1_front.eval()
        self.critic2_front.eval()

    def get_action(self, ob: torch.Tensor, record_stats: bool = True) -> torch.Tensor:
        """
        线程安全的动作生成（使用前台网络）
        :param ob: 观测 (batch_size, ob_dim)
        :param record_stats: 是否记录动作统计信息
        :return: 动作 (batch_size, ac_dim)
        """
        with torch.no_grad():
            action, std, log_prob = self.actor_front(ob)

        # 降低动作统计的记录频率
        if record_stats and self.global_step % self.log_frequency == 0:
            action_mean = action.mean().item()
            action_max = action.max().item()
            action_min = action.min().item()
            action_std = action.std().item()
            log_prob_mean = log_prob.mean().item()

            self.logger.log("Action/mean", action_mean, self.global_step)
            self.logger.log("Action/max", action_max, self.global_step)
            self.logger.log("Action/min", action_min, self.global_step)
            self.logger.log("Action/std", action_std, self.global_step)
            self.logger.log("Action/log_prob", log_prob_mean, self.global_step)

        return action

    def update(self,
               ob: torch.Tensor,
               ac: torch.Tensor,
               nex_ob: torch.Tensor,
               rew: torch.Tensor,
               done: torch.Tensor) -> dict:
        """
        后台网络更新（线程安全）
        :param ob: 观测 (batch_size, ob_dim)
        :param ac: 动作 (batch_size, ac_dim)
        :param nex_ob: 下一观测 (batch_size, ob_dim)
        :param rew: 奖励 (batch_size,)
        :param done: 终止标志 (batch_size,)
        :return: 损失字典
        """
        # 数据移到设备上
        ob = ob.to(self._device)
        ac = ac.to(self._device)
        nex_ob = nex_ob.to(self._device)
        rew = rew.to(self._device).unsqueeze(1)  # (batch_size, 1)
        done = done.to(self._device).unsqueeze(1)  # (batch_size, 1)

        # 只在特定步数记录日志，减少开销
        should_log = self.global_step % self.log_frequency == 0

        # 记录奖励统计（仅批量均值）
        if should_log:
            rew_mean = rew.mean().item()
            self.logger.log("Reward/batch_mean", rew_mean, self.global_step)

        # 加锁保护后台网络更新（同一时间只有一个线程更新）
        with self._lock:
            # 1. 计算目标Q值（使用前台网络，保证一致性）
            with torch.no_grad():
                nex_ac, nex_log_prob = self.actor_front(nex_ob)
                target_q1 = self.critic1_front(nex_ob, nex_ac)
                target_q2 = self.critic2_front(nex_ob, nex_ac)
                target_q = torch.min(target_q1, target_q2) - self.alpha * nex_log_prob
                target_q = rew + self.gamma * (1 - done) * target_q

            # 记录目标Q值统计
            if should_log:
                self.logger.log("Q_value/target_mean", target_q.mean().item(), self.global_step)
                self.logger.log("Q_value/target_std", target_q.std().item(), self.global_step)

            # 2. 更新 Critic 网络
            current_q1 = self.critic1_back(ob, ac)
            current_q2 = self.critic2_back(ob, ac)
            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            critic1_norm = torch.nn.utils.clip_grad_norm_(self.critic1_back.parameters(), max_norm=1.0)
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            critic2_norm = torch.nn.utils.clip_grad_norm_(self.critic2_back.parameters(), max_norm=1.0)
            self.critic2_optimizer.step()

            # 记录Critic统计
            if should_log:
                self.logger.log("Loss/critic1", critic1_loss.item(), self.global_step)
                self.logger.log("Loss/critic2", critic2_loss.item(), self.global_step)
                self.logger.log("Gradients/critic1_norm", critic1_norm.item(), self.global_step)
                self.logger.log("Gradients/critic2_norm", critic2_norm.item(), self.global_step)
                self.logger.log("Q_value/current1_mean", current_q1.mean().item(), self.global_step)
                self.logger.log("Q_value/current2_mean", current_q2.mean().item(), self.global_step)

            # 3. 更新 Actor 网络（冻结Critic梯度）
            for param in self.critic1_back.parameters():
                param.requires_grad = False
            for param in self.critic2_back.parameters():
                param.requires_grad = False

            actor_ac, actor_log_prob = self.actor_back(ob)
            q1 = self.critic1_back(ob, actor_ac)
            q2 = self.critic2_back(ob, actor_ac)
            q = torch.min(q1, q2)
            actor_loss = - (q - self.alpha * actor_log_prob).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(self.actor_back.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # 解冻Critic梯度
            for param in self.critic1_back.parameters():
                param.requires_grad = True
            for param in self.critic2_back.parameters():
                param.requires_grad = True

            # 记录Actor统计
            if should_log:
                self.logger.log("Loss/actor", actor_loss.item(), self.global_step)
                self.logger.log("Gradients/actor_norm", actor_norm.item(), self.global_step)
                self.logger.log("Actor/entropy", -actor_log_prob.mean().item(), self.global_step)

            # 更新计数器和全局步数
            self._update_count += 1
            self.global_step += 1

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "step": self.global_step
        }

    def flush(self) -> None:
        """
        线程安全的网络切换（原子指针替换）
        1. 快速替换前台网络指针（原子操作，无阻塞）
        2. 用新前台网络同步旧前台网络参数（后台网络可继续训练）
        """
        # 仅当更新次数达到阈值时执行
        if self._update_count < self.target_update_freq:
            return

        # 步骤1：原子替换前台网络指针（关键！仅替换引用，O(1)时间）
        # Python中赋值操作是原子的，无需额外锁
        old_actor_front = self.actor_front
        old_critic1_front = self.critic1_front
        old_critic2_front = self.critic2_front

        # 用训练好的后台网络替换前台网络
        self.actor_front = copy.deepcopy(self.actor_back)
        self.critic1_front = copy.deepcopy(self.critic1_back)
        self.critic2_front = copy.deepcopy(self.critic2_back)

        # 新前台网络设为eval模式
        self.actor_front.eval()
        self.critic1_front.eval()
        self.critic2_front.eval()

        # 步骤2：同步旧前台网络参数（非原子，但不影响主线程）
        # 这一步是为了后续软更新，旧网络已不再被get_action使用，无需加锁
        self._soft_update(old_actor_front, self.actor_front)
        self._soft_update(old_critic1_front, self.critic1_front)
        self._soft_update(old_critic2_front, self.critic2_front)

        # 重置更新计数器
        self._update_count = 0

    def _soft_update(self, target_net: nn.Module, source_net: nn.Module) -> None:
        """软更新网络参数：target = tau * source + (1 - tau) * target"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def close(self):
        """清理资源"""
        self.logger.stop()

    def to(self, device: torch.device) -> "DoubaoSACPolicy":
        """重写to方法，确保所有网络都移到指定设备"""
        super().to(device)
        self._device = device

        # 移动所有网络
        self.actor_front.to(device)
        self.critic1_front.to(device)
        self.critic2_front.to(device)

        self.actor_back.to(device)
        self.critic1_back.to(device)
        self.critic2_back.to(device)

        return self


class Actor(nn.Module):
    """SAC Actor网络（输出动作分布）"""

    def __init__(self, ob_dim: int, ac_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(ob_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, ac_dim)
        self.log_std_layer = nn.Linear(hidden_dim, ac_dim)

        # 动作范围限制（默认[-1,1]，可根据环境调整）
        self.action_scale = torch.tensor(10.0)
        self.action_bias = torch.tensor(0.0)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, ob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ob: 观测 (batch_size, ob_dim)
        :return: (动作, 对数概率) -> (batch_size, ac_dim), (batch_size, 1)
        """
        x = F.relu(self.fc1(ob))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # 重参数化采样
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # 计算对数概率（考虑tanh变换）
        log_prob = normal.log_prob(x_t)
        # 修正tanh带来的概率密度变化
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, std, log_prob


class Critic(nn.Module):
    """SAC Critic网络（Q函数）"""

    def __init__(self, ob_dim: int, ac_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(ob_dim + ac_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, ob: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        """
        :param ob: 观测 (batch_size, ob_dim)
        :param ac: 动作 (batch_size, ac_dim)
        :return: Q值 (batch_size, 1)
        """
        x = torch.cat([ob, ac], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# 使用示例
if __name__ == "__main__":
    # 初始化参数
    ob_dim = 10
    ac_dim = 2
    batch_size = 32

    # 创建policy实例（降低日志频率）
    policy = DoubaoSACPolicy(ob_dim=ob_dim, ac_dim=ac_dim, log_frequency=10)


    # 模拟主线程：生成动作
    def main_thread():
        for _ in range(100):
            ob = torch.randn(batch_size, ob_dim).to(policy._device)
            action = policy.get_action(ob)
            assert action.shape == (batch_size, ac_dim)
            print(f"Main thread: Generated action shape {action.shape}")
            time.sleep(0.01)  # 模拟环境交互延迟


    # 模拟后台线程：更新网络
    def background_thread():
        for _ in range(100):
            # 生成模拟数据
            ob = torch.randn(batch_size, ob_dim)
            ac = torch.randn(batch_size, ac_dim)
            nex_ob = torch.randn(batch_size, ob_dim)
            rew = torch.randn(batch_size)
            done = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

            # 更新网络
            losses = policy.update(ob, ac, nex_ob, rew, done)
            print(f"Background thread: Step {losses['step']}, Losses {losses}")

            # 每10次更新执行一次flush
            if policy._update_count >= policy.target_update_freq:
                policy.flush()
                print("Background thread: Flushed networks")

            time.sleep(0.02)  # 模拟训练延迟


    # 启动线程
    main_thread = threading.Thread(target=main_thread)
    background_thread = threading.Thread(target=background_thread)

    main_thread.start()
    background_thread.start()

    main_thread.join()
    background_thread.join()

    # 清理资源
    policy.close()
    print("Training completed. Logs saved to:", policy.logger.writer.log_dir)