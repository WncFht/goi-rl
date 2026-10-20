import numpy as np
import torch

from my_lidar_renderer import UltraFastLidarRenderer
from my_logger import Logger
from my_replay_buffer import ThreadSafeReplayBufferTorch


class DataManager:
    def __init__(self, recv_bs, recv_dim, ob_dim, ac_dim,
                 actor,
                 buffer: ThreadSafeReplayBufferTorch,
                 logger: Logger,
                 device):  # recv_dim=39,前32是低维信息，后7是渲染相关信息
        self.recv_bs = recv_bs
        self.recv_dim = recv_dim
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.device = device

        self.actor = actor
        self.buffer = buffer
        self.logger = logger

        self.lidar_renderer = UltraFastLidarRenderer("assets/related_seg_idxs.pt")

        ###我在这里加上雷达渲染，这样保存的就有点大了，不过没关系。

        self.is_initial = True
        self.last_frame_np = np.zeros((recv_bs, recv_dim), dtype=np.float32)
        self.new_frame_np = np.zeros((recv_bs, recv_dim), dtype=np.float32)
        self.last_frame_props_np = np.zeros(6, dtype=np.float32)
        self.new_frame_props_np = np.zeros(6, dtype=np.float32)
        self.last_ac_np = np.zeros((recv_bs, ac_dim), dtype=np.float32)
        self.new_ac_np = np.zeros((recv_bs, ac_dim), dtype=np.float32)
        self.last_ob = None
        self.last_ac = None
        self.new_ob = torch.zeros((recv_bs, ob_dim), device=device)  # torch随便一个两个括号都行；np只能两个括号
        self.new_ac = torch.zeros((recv_bs, ob_dim), device=device)

        self.ob_std = np.array([[  # 默认ob_dim=32
            1., 1.,  # [0:2] 上次的鼠标输入
            1., 1., 1., 1.,  # [2:6] 光标位置速度
            1., 1.,  # [6:8] 玩家绝对速度xy
            30., 400.,  # [8:10] 玩家角度，角速度绝对速度xy
            1., 1., 400.,  # [10:13] 旋转关节转角sin cos，角速度
            1., 800.,  # [13:15] 直线关节的行程和速度
            1., 1., 1., 1.,  # [15:19] handle和pole_middle的相对位置
            1., 1., 1., 1., 20., 20.,  # [19:25] tip的相对位置、角度sc、速度xy
            1., 1., 1., 1.,  # [25:29]锤子碰撞冲量矢量xy，缸碰撞冲量矢量xy
            1., 1., 500.,  # [29:32]前进切向xy，前进距离最远到860左右，我随便给个500。 第31位是进度。
        ]], dtype=np.float32)  # 第32位：活着的计时时间，33:36玩家xyr，36:39锤子xyr。

    # 收到一帧之后的调用顺序：先get_ob_from_frame从当前帧获得ob，然后马上根据ob get_action，然后action发出去，最后再打包放入缓冲区。

    def set_frame_with_lock(self, frame: list):
        self.new_frame_np.flat = frame[:-6]
        self.new_frame_props_np.flat = frame[-6:]

    def get_new_ob_ac_from_frame(self) -> np.ndarray:

        self.new_ob = torch.zeros((self.recv_bs, self.ob_dim), device=self.device)

        ##########low_dim ob
        self.new_ob[:, :32] = torch.as_tensor(self.new_frame_np[:, :32] / self.ob_std, device=self.device)

        ########## 此处直接在这里渲染出high_dim ob如scanner，pic等。
        distances = self.lidar_renderer(torch.as_tensor(self.new_frame_np[:, 33:35], device=self.device),
                                        self.new_frame_props_np[:3], self.new_frame_props_np[3:])
        self.new_ob[:, 32:] = distances / 10  # 距离归一化到0到2

        self.new_ac = self.actor.get_action(self.new_ob)
        self.new_ac_np = self.new_ac.cpu().numpy()
        return self.new_ac_np  # 转回CPU供环境使用

    def calc_reward(self, frame1: np.ndarray, ac: np.ndarray, frame2: np.ndarray) -> dict[str, np.ndarray]:
        # ====计算一条transition的奖励：利用 frame1 ac frame2 来计算这一条transition的奖励。
        # frame的定义为0到31维见上，32是活着的时间计时，33:36玩家xyr，36:39锤子xyr。
        # 1.基础奖励为进度（frame第31位）差值。现在进度已经限制了单帧摆率0.4了。奖励单帧最大到2。#先不惩罚大退步太猛？后面慢慢惩罚。
        # 这里可以整成类似课程学习的思路，退步比较多的时候就往少了clip，退步比较少的时候就放宽clip或者不clip
        rew = (frame2[:, 31] - frame1[:, 31]).clip(min=-0.2)

        # 奖励“人的速度向锤子冲量方向的投影”，我们选只留正值，对于负值不惩罚。最大0.5，速度10米每秒时达到。
        ham_impulse_player_vel_rew = (
                (frame2[:, 6] * frame2[:, 25] + frame2[:, 7] * frame2[:, 26]) * np.float32(0.05)).clip(0, 0.5)
        # 奖励速度向上，不惩罚向下
        up_vel_rew = (frame2[:, 7] * np.float32(0.05)).clip(0, 0.5)

        # 我直接惩罚锤子受到向下的冲量，而且用比较严厉的惩罚。而向上并不值得奖励。然后从头开始训。冲量一次最多直接扣1。而且跟方向有关，纯下最扣
        down_impulse_pun = (frame2[:, 26] * np.float32(1.0)).clip(-1, 0)

        # 辅助奖励3：惩罚动作范数，最大范数20要扣0.2，所以是乘以0.01
        action_l2_pun = np.linalg.norm(ac, axis=1) * np.float32(-0.01)

        rew_dict = {
            "reward/prog_rew": rew,
            "reward/ham_impulse_player_vel_rew": ham_impulse_player_vel_rew,
            "reward/up_vel_rew": up_vel_rew,
            "reward/down_impulse_pun": down_impulse_pun,
            "reward/action_l2": action_l2_pun,
        }
        return rew_dict

    # 把UDP帧打包成ob,ac,nex_ob,rew,done.推到数据集里。注意，其实ob和ac是旧ob和旧ac；新ob和新ac推完之后才能替换成旧ob和旧ac。
    def pack_frame(self):
        # 其实这里主要是rew和done的计算工作。新旧ob都已经有了，而且都是torch的。
        # ====处理reward。
        rew_dict = self.calc_reward(self.last_frame_np, self.last_ac_np, self.new_frame_np)
        rew = np.sum(list(rew_dict.values()), axis=0)  # 多项长num_duplis的奖励变成一项长num_duplis的奖励

        # ====处理done。
        # 第32位：活着的计时时间，它是单调的。new_frame_np[32]是时间值，如果减小了说明done了。
        # 对于done了当中的情况，有输，赢，超时三种情况。取决于新frame模4的尾数。
        done = self.new_frame_np[:, 32] < self.last_frame_np[:, 32]  # last_frame_np 有可能是全0，但是已经设了全0不推buffer。
        done_condition = self.new_frame_np[:, 32].astype(np.int32) & 3
        lose_condition = np.logical_and(done, done_condition == 0)
        win_condition = np.logical_and(done, done_condition == 1)
        rew[lose_condition] = -20  # done了且当前尾数是0表示输了
        rew[win_condition] = 20  # done了且当前尾数是1表示赢了

        terminate_condition = np.logical_or(lose_condition, win_condition)
        # 不用修复done了之后的next_ob跳变的问题，因为算奖励用的是 (1 - done) * target_q，如果done是1，那么next_ob的值不会影响估值！
        keep_np = np.logical_or(np.logical_not(done), terminate_condition)
        keep = torch.as_tensor(keep_np)  # 扔掉done而不terminate的帧，因为这些帧的transition不对。
        num_frames = keep.sum().item()

        if num_frames and self.last_ob is not None:  # 这是能记录的条件。
            # 在tensorboard中记录奖励，只记录keep的那些，而且win了或者lose了的也先不记。
            rew_dict = {k: v[keep_np].sum().item() for k, v in rew_dict.items()}

            # 因为第一帧的旧ob和旧ac全是none，所以不能往里推。
            self.buffer.push(self.last_ob[keep],
                             self.last_ac[keep],
                             self.new_ob[keep],
                             torch.as_tensor(rew, device=self.device)[keep],
                             torch.as_tensor(done, device=self.device)[keep])
        else:  # 没有数据，返回空字典。
            rew_dict = {k: 0 for k, v in rew_dict.items()}

        self.last_frame_np[:] = self.new_frame_np  # 此处不能复制引用，否则会被改动，必须值覆盖。
        self.last_frame_props_np[:] = self.new_frame_props_np
        self.last_ac_np = self.new_ac_np  # 此处可以复制引用。因为ac是new出来的，不会值覆盖。
        self.last_ob = self.new_ob
        self.last_ac = self.new_ac

        return num_frames, rew_dict
