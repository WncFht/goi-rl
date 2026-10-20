##首先整一个只保留最近一帧的UDP记录器,，放在辅助线程里面。
import socket
import struct
import time
import torch
import winreg
import signal
import threading
import numpy as np


from claude_sac_policy import SACPolicy
from my_data_manager import DataManager
from my_replay_buffer import ThreadSafeReplayBufferTorch
from my_udp_receiver import UDPReceiver
from my_logger import Logger


def update_policy(policy: SACPolicy, buffer: ThreadSafeReplayBufferTorch, recorder: UDPReceiver, batch_size=1024):
    while buffer.current_size < batch_size * 10:
        print("waiting for enough initial data...")
        time.sleep(5)

    print("训练线程启动！")

    while recorder.running:
        for _ in range(10):
            ob, ac, next_ob, rew, done = buffer.sample(batch_size)
            policy.update(ob, ac, next_ob, rew, done)
        policy.flush()
        print("策略更新！")
        time.sleep(1)

def main():
    # 读取当前配置的复制体数量
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Bennett Foddy\Getting Over It", 0, winreg.KEY_READ)
    num_duplis, _ = winreg.QueryValueEx(key, "numDuplis_h1216609508")
    winreg.CloseKey(key)
    floats_per_dupli = 39
    num_floats = num_duplis * floats_per_dupli + 6
    print(f"复制体数量：{num_duplis}，单帧浮点数量{num_floats}")

    ob_dim = 32 + 128 # 32 low_dim + 128 lidar
    ac_dim = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 创建强化学习所用的缓冲区
    buffer = ThreadSafeReplayBufferTorch(ob_dim=ob_dim, ac_dim=ac_dim, device=device)

    # 创建对应数量的录制器
    recorder = UDPReceiver(num_floats)
    receiver_thread = threading.Thread(target=recorder.record_udp_data, args=('localhost', 12346))  # 用子线程启
    receiver_thread.start()

    # 创建策略
    policy = SACPolicy(ob_dim=ob_dim, ac_dim=ac_dim, device=device)
    trainer_thread = threading.Thread(target=update_policy, args=(policy, buffer, recorder))
    trainer_thread.start()

    # 创建动作发送器
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    #创建数据记录器
    logger = Logger()

    #创建数据管理器
    data_manager=DataManager(num_duplis,floats_per_dupli,ob_dim=ob_dim, ac_dim=ac_dim,
                             actor=policy,buffer=buffer,logger=logger, device=device)

    interval = 1.0 / 40  # 0.025 秒 = 25 毫秒
    infer_cnt = 0  # 40Hz推理的次数
    old_timestamp = 0
    start_time = forty_frames_timer = time.perf_counter()
    sleep_max, sleep_min, sleep_sum, sleep_sum2, fps = 0, interval * 1000, 0, 0, 0
    all_rew_dict=None

    while recorder.running:  # ctrlC已经被信号捕获了，所以这里不会有KeyboardInterrupt
        with recorder.lock:
            if recorder.timestamp > old_timestamp:  # 看下有没有新值过来，如果有就直接干，如果没有就跳过这次循环了，等下次40ms。
                data_manager.set_frame_with_lock(recorder.frame)
            else:
                start_time = time.perf_counter()  # 如果时间到了新数还没来，就稍微把时序变松让一让
                continue
        infer_cnt += 1
        ac = data_manager.get_new_ob_ac_from_frame()
        sender.sendto(ac.tobytes(), ('localhost', 12345))
        f, rew_dict = data_manager.pack_frame()

        fps += f
        if all_rew_dict:
            for k in all_rew_dict.keys(): all_rew_dict[k]+=rew_dict[k]
        else:
            all_rew_dict = rew_dict


        ##############以下是统计一些时序和统计量的情况。我现在以上次收到udp的时间点为基准，延迟25ms作为下一次查看udp值和推理的时间点。
        #########如果从来就没有收到过UDP，那就一直在前面绕着进不到统计量这里来。
        next_time = start_time + interval
        sleep_time = next_time - time.perf_counter()
        start_time = next_time
        if sleep_time > 0.01:
            time.sleep(sleep_time - 0.01)
            while time.perf_counter() < next_time: pass  # 如果剩的时间不到10ms的时候就不要把CPU让出去了。这个定时更精确。

        sleep_time *= 1000  # 下面以毫秒为单位
        if sleep_max < sleep_time: sleep_max = sleep_time
        if sleep_min > sleep_time: sleep_min = sleep_time
        sleep_sum += sleep_time
        sleep_sum2 += sleep_time ** 2
        if infer_cnt % 40 == 0:
            logger.log_dict_collect({k:v/fps for k, v in all_rew_dict.items()},infer_cnt)
            all_rew_dict=None #每40帧聚合一次结果，然后清空结果。

            forty_frames_time_interval = time.perf_counter() - forty_frames_timer
            forty_frames_timer = time.perf_counter()
            sleep_mean = sleep_sum / 40
            sleep_variance = (max(0, sleep_sum2 / 40 - sleep_mean ** 2)) ** 0.5

            print(f"最近40帧用了{forty_frames_time_interval:.2f}s, "
                  f"推理时间裕度max{sleep_max:.2f}ms, min{sleep_min:.2f}ms, mean{sleep_mean:.2f}ms, var{sleep_variance:.2f}ms, 收集帧数{fps}, 帧率{fps / forty_frames_time_interval:.2f}, ")
            sleep_max, sleep_min, sleep_sum, sleep_sum2, fps= 0, interval * 1000, 0, 0, 0

    logger.close()
    receiver_thread.join()  # 避免主线程退出太快子线程还没退完
    trainer_thread.join()


if __name__ == "__main__":
    main()
