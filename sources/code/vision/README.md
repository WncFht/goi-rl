# GoI-RL-Vision

基于视觉的强化学习，训练 AI 玩 *Getting Over It with Bennett Foddy*（掘地求升）。

**Route C 混合方案**: 屏幕截帧 (DXCam) → CNN 策略 → 鼠标控制 + 内存读取 camera Y 做奖励信号。

不需要修改游戏本体（参照项目 [Heuriz/goi-rl](https://gitee.com/Heuriz/goi-rl) 需要 mod 游戏输出 UDP 物理数据），只需要一个 BepInEx 插件输出相机坐标。

## 硬件

| 组件 | 型号 |
|------|------|
| CPU | Intel i9-14900K |
| GPU | NVIDIA RTX 4090 D (24 GB) |
| OS | Windows 11 Pro |
| Python | 3.12.4 (Anaconda) |

## 架构

```
src/
├── game_interface/       # 游戏交互层
│   ├── capture.py        # DXCam GPU 屏幕截帧
│   ├── mouse.py          # SendInput 鼠标注入 + 光标锁定
│   ├── window.py         # Win32 窗口管理
│   └── game_launcher.py  # 游戏进程启动/关闭
│
├── state_reader/         # 状态读取层
│   ├── camera_reader.py  # 从 BepInEx 插件文件读取 camera (x,y,z)
│   ├── memory_reader.py  # pymem 内存读取 + 注册表回退
│   └── address_finder.py # Cheat Engine 地址发现指南
│
├── environment/          # RL 环境 (Gymnasium)
│   ├── goi_env.py        # 主环境: 截帧 → 奖励 → 动作 → 循环
│   ├── observation.py    # 帧堆叠 (4×84×84 灰度, channel-first)
│   ├── reward.py         # 7 组件不对称奖励函数
│   ├── action.py         # 动作去归一化 [-1,1] → 像素 delta
│   └── register.py       # Gymnasium 环境注册
│
├── training/             # 训练
│   ├── train.py          # PPO 训练主入口 (热键控制 + TensorBoard)
│   ├── evaluate.py       # 确定性评估 + 指标
│   └── callbacks.py      # 检查点/Metrics/崩溃恢复/停止标志回调
│
└── analysis/             # 分析 (待完善)
    ├── plot_training.py  # 训练曲线绘制
    └── replay_episode.py # Episode 回放
```

## 各文件详解

### game_interface/ — 游戏交互

| 文件 | 功能 | 状态 |
|------|------|------|
| `capture.py` | DXCam GPU 截取游戏窗口区域，输出灰度 numpy 数组。20 FPS 独立线程 | 正常 |
| `mouse.py` | `SendInput` 发送相对鼠标移动让 Unity `Input.GetAxis("Mouse X/Y")` 读取。训练时自动锁光标到右下角 | 正常 |
| `window.py` | 查找游戏窗口句柄，获取窗口区域坐标 | 正常 |
| `game_launcher.py` | 启动/关闭游戏进程 | 未测试 |

### state_reader/ — 状态读取

| 文件 | 功能 | 状态 |
|------|------|------|
| `camera_reader.py` | 读取 BepInEx CameraLogger 插件输出的 `camera_y.txt`，获取实时相机 (x,y,z) 坐标。**当前实际使用** | 正常 |
| `memory_reader.py` | 通过 pymem 读取游戏内存中的相机 Y 值，带注册表回退。需要 Cheat Engine 找到地址 | 未配置地址 |
| `address_finder.py` | Cheat Engine Mono Dissect 地址查找指南 | 文档 |

### environment/ — RL 环境

| 文件 | 功能 | 状态 |
|------|------|------|
| `goi_env.py` | Gymnasium Env。10Hz 步频，动作空间 2D 连续 [-1,1]，观测 4×84×84 灰度。含黑屏死亡检测 | 正常 |
| `observation.py` | 帧堆叠处理器，4 帧灰度 84×84 channel-first | 正常 |
| `reward.py` | **7 组件不对称奖励** (详见下方) | 持续调优 |
| `action.py` | 去归一化 + 时间平滑 | 正常 |

### training/ — 训练

| 文件 | 功能 | 状态 |
|------|------|------|
| `train.py` | 主训练入口。自动启动 TensorBoard，热键控制 (Ctrl+Shift+S 开始, Ctrl+Shift+Q 停止)，SB3 ep_info_buffer 补丁 | 正常 |
| `callbacks.py` | 检查点保存 + 指标日志 + 崩溃恢复 + StopFlagCallback | 正常 |
| `evaluate.py` | 加载模型确定性评估 | 未测试 |

## 奖励函数 (7 组件)

参照 B 站 UP [茹莺](https://space.bilibili.com/442902038) 的开源项目 [Heuriz/goi-rl](https://gitee.com/Heuriz/goi-rl) 设计，适配纯 camera Y 观测。

| # | 组件 | 说明 |
|---|------|------|
| ① | 进度奖励 | 上升: `height_delta × 10`，下降: clip 到 -0.2（非对称，为 swinging 留空间） |
| ② | 向上奖励 | 任何上升额外 +50%，cap 1.0 |
| ③ | 跌落惩罚 | 从峰值跌超 3.0 时按比例罚 |
| ④ | 卡住惩罚 | 100 步无高度变化 → -0.1/步 |
| ⑤ | 地面卡住 | y ≈ 地面 + 30 步不前进 → -0.3/步（惩罚"锤子不接地+缸接地"的无用状态） |
| ⑥ | 离地上升 | y > 地面 2.0 且上升 → 额外 +30%（间接奖励"锤子锚住+缸离地"） |
| ⑦ | 动作代价 | `-0.01 × ‖action‖²` |

另：黑屏死亡（掉水里） -5，前 30 步有保护期免罚。

## 已知问题 / 待优化

### 🔴 高优先级

1. **纯 camera Y 奖励无法教代理"怎么用锤子"**  
   参照项目有 UDP 物理数据（玩家速度、锤子碰撞向量等），可以做"锤-人速度对齐"奖励直接指导锤子使用。我们只有结果（高度变化），没有过程信号。代理必须靠运气发现有效动作。

2. **CNN 策略可能对 2D 控制任务过大**  
   当前用 Nature DQN CNN 处理 4×84×84 灰度帧来输出 2 个连续值。这个任务可能不需要 CNN——直接用 camera Y 数值 + MLP 或许更快收敛。

3. **单环境，数据效率低**  
   参照用 40 个并行环境 + SAC（off-policy），我们只有 1 个环境 + PPO（on-policy）。每 wall-clock 小时的有效数据量差 100+ 倍。

4. **episode_reward_mean 仍为负值** (~-4.2)  
   最好的 episode 才接近不亏 (-0.08)，平均还在亏。死亡惩罚和动作代价主导了前期训练。

### 🟡 中优先级

5. **MetricsCallback 不记录奖励子组件**  
   TensorBoard 看不到 `reward/prog`、`reward/aerial_bonus` 等分项曲线，难以调试奖励设计。

6. **DXCam 实例重复创建的警告**  
   `DXCamera instance already exists` — 不影响功能但有噪音。需要在 episode 重开时复用实例。

7. **game_launcher.py 未测试**  
   目前需要手动打开游戏。

8. **evaluate.py 未测试**  
   没有跑过确定性评估。

9. **CameraLogger 插件需要 BepInEx**  
   需要手动安装 BepInEx + 放入 `plugins/CameraLogger.dll`。没有这个插件就没有 camera Y 读数。

### 🟢 低优先级

10. **analysis/ 模块是空壳**  
    `plot_training.py` 和 `replay_episode.py` 只有骨架代码。

11. **Cheat Engine 地址文件未填充**  
    `config/memory_addresses.json` 只有一个占位地址，需要通过 Cheat Engine 找到真实指针链后 pymem 才能用。

12. **训练超参未经调优**  
    PPO 的 `ent_coef`、`n_steps`、`learning_rate` 等用的是默认值，没有做 hyperparameter sweep。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练（需要先打开游戏 + 安装 BepInEx CameraLogger 插件）
python -m src.training.train

# 从 checkpoint 续训
python -m src.training.train --resume checkpoints/goi_ppo_final.zip

# TensorBoard
tensorboard --logdir logs
```

**热键**: `Ctrl+Shift+S` 开始训练，`Ctrl+Shift+Q` 停止。

## 依赖

- stable-baselines3 (PPO)
- gymnasium
- PyTorch (CUDA)
- DXCam (GPU 截帧)
- OpenCV
- pymem (内存读取)
- pywin32
- keyboard (全局热键)
- tensorboard
- pyyaml

## 参考

- [Heuriz/goi-rl](https://gitee.com/Heuriz/goi-rl) — 参照实现 (mod 游戏 + UDP + SAC + CUDA LIDAR + 40 并行环境)
- 茹莺 B 站系列 "掘地求升AI进展" (9 集)
