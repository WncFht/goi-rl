# 我用hopper v5作为训练对象吧，看看能不能复现以前sac的效果。主要是因为hopper和halfcheeta相比，hopper可能会提前终止。
# 我先做一个必须固定以40Hz交互的环境示例。出32个并行环境。

# 现在总共需要两个线程。
# 一个40Hz线程：读ob，放入上一次的buf，调用网络，推出ac，收新ob，放入这一次的buf。
# 一个训练循环：从buf抽取一批，训练，更新网络*10步，写出网络参数。
# 所以总共就这两个线程。

# 第一步我先把agent建起来吧。。反正就是一个actor，一个critic。我们会发现critic是不用跨线程的，只有actor跨线程。


import torch
from numpy import ndarray
from torch import nn, Tensor, distributions
from replay_buffer import ThreadSafeReplayBufferTorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, _device: torch.Device) -> nn.Module:
    layers = []
    current_dim = input_dim
    for o_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, o_dim))
        layers.append(nn.ELU())
        current_dim = o_dim
    layers.append(nn.Linear(current_dim, output_dim))

    return nn.Sequential(*layers).to(_device)


class MySAC(nn.Module):
    def __init__(self, ob_dim, ac_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ob_dim: int = ob_dim
        self.ac_dim: int = ac_dim

        self.num_actor_samples: int = 1
        self.num_critic_updates: int = 1
        self.discount: float = 0.99
        self.temperature: float = 0.1

        self.critics = torch.nn.ModuleList(  # 两个critic，使用minq
            [build_mlp(ob_dim + ac_dim, [512, 256, 128], 1, self.device) for _ in range(2)])

    def update_critic(self, ob: Tensor, ac: ndarray, nex_ob: Tensor, rew: Tensor, done: Tensor):
        # 自举更新Q函数
        # 第一步，计算新Q（即新状态+新动作的网络估值），这一步必须无梯度。

        # 第二步，新Q折扣加奖励，作为老Q的学习目标。如果done了，那么老Q的学习目标就是纯奖励。

        pass

    def update(self, ob: Tensor, ac: ndarray, nex_ob: Tensor, rew: Tensor, done: Tensor):
        # 第一步，更新critic（Q函数自举更新，由于我们的数据不成序列所以只能单步自举打移动靶更新）
        for _ in range(self.num_critic_updates):
            self.update_critic(ob, ac, nex_ob, rew, done)

        # 第二步，更新actor。


class MyActorNet(nn.Module):
    def __init__(self, ob_dim, ac_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ob_dim: int = ob_dim
        self.ac_dim: int = ac_dim
        self.mean_net = build_mlp(ob_dim, [512, 256, 128], ac_dim, self.device)
        self.log_std = nn.Parameter(torch.full((ac_dim,), 0.0, dtype=torch.float32, device=self.device))

    def forward(self, obs: torch.Tensor) -> distributions.Distribution:
        mean





class MySACOnlyActor(nn.Module):

    def __init__(self):
        super().__init__()

# 我们一般认为ob_dim在push进去的时候是展平的，所以用int不用tuple。
# ob_dim=47
# ac_dim=2

# #我现在需要一个replay buffer。训练循环是每次从replay buffer里面取出来训练。
# buffer=ThreadSafeReplayBufferTorch(ob_dim,ac_dim)
#
# actor=build_mlp(ob_dim,[512,256,128],3,device)
# critic=build_mlp(ob_dim+ac_dim,[512,256,128],3,device)
# critic_optimizer=torch.optim.Adam(critic.parameters(),lr=1e-3)
#
# def update_critic(ob: Tensor, ac: Tensor, rew: Tensor, nex_ob: Tensor, done: Tensor) -> None:
#     pass
#
# def update_actor(ob: Tensor) -> None:
#     pass

# 慢线程：推理线程，可以这样，有自己的40Hz时钟，时钟到了立刻取最新的obs，然后马上推理，推理完结果马上发出。
# 搞定了这一切之后，再存sars'数据。存数据做法：
# 如果没有旧sa，则当前sa变成旧sa。
# 如果有旧sa，则旧sa和当前s联合，作为一帧数据存进去。然后当前s和a占据了原先的旧sa的位置。

# 需要扔数据的情况：s里面有一个存活时间维度，正常单调递增。如果死了重开或者截断重开则归零继续递增。
# 这样就会有一个旧s数大新s数小的情况。这一帧是扔还是怎么着呢？？好像不扔也行。这样吧，不扔了，这一帧直接s'和s存一样的，然后done掉。然后新s留着下次存。
# 如果是truncated了呢，比如活到120*40=4800就截断随机传送了。这也会导致收到一个旧s数大新s数小的帧。这时怎么办呢：这个帧肯定不能要了。所以这条数据丢掉就行了。
# 而且我们是用sac训练的时候，其实不关心是否truncated，只关心是否die，所以扔掉truncated的那一帧（只要不是die就行）是没问题的。
# 现在往缓冲区推数据的代码已经支持非整64推送了，所以没问题。
# 但是，现在就怕一种特别寸的情况：本来在该截断的那一瞬间，但是正好die了。这个时候我应该怎么区分它是截断了还是die了呢。。。

# 我得重新梳理一下。。。一个是存活时间t的变化，一个是进度p的变化。这两个东西共同决定了是die了还是截断了还是怎么样。
# 如果是正常：t单调递增。进度p无所谓。t最多比如说允许增到4800。
# 如果是die了：在C#里面一旦检测到进度p退步过大，会直接触发时钟重置和随机传送，发回来的状态直接是0/1/2时钟和新的状态。
# 如果是截断了：C#里面检测到t达到了4800，会直接触发时钟重置和随机传送，发回来的状态直接是0/1/2时钟和新的状态。
# 好像根据发回来的东西确实没办法判断上一回合的时钟重置是die了导致的还是时间到了导致的。
# 那我可以这样，用一个小trick：时钟t每次+2而不是+1，这样保证奇偶性。
# 如果是die了导致的触发，让下次的时钟一直为奇数；如果是截断导致的触发则让下次的时钟一直为偶数。
# 这样在python里，如果检测到时间非单调跳变且为奇数，则存上一帧的时候存s,a,r=-10，还是旧s，done=1。然后旧sa变量存新sa。
# 如果时间非单调跳变且为偶数，则应该扔掉上一帧不存，然后然后旧sa变量存新sa。这样就对了。


# #快线程：训练线程，不需要与环境交互，可以一直抽样训练抽样训练...
# while True:
#     a_batch=buffer.sample() # ob,ac,rew,nex_ob,done
#     update_critic(*a_batch) #我直接取出一批数据更新critic。
#     update_actor(a_batch[0]) #更新actor只需要ob。
#
# #actor需要两个，actor直接顺着Q网络做梯度上升就行了。
# #critic因为clipQ所以也需要两个Q网络，用两个网络之中较小的那个输出值来算Q网络的更新target。
#
# exit(0)
#
# import gymnasium as gym
# env = gym.make('Hopper-v5') #,render_mode="human"  #3维输入，11维输出
# obs, info = env.reset()
# print(obs,info)
# action = env.action_space.sample()
# print(action)
# obs, reward, done, truncated, info = env.step(action)
# print(obs, reward, done, truncated, info)
#
# while True:
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     if done:
#         env.reset()
#
#
# env.close()
