import socket
import struct
import time
import winreg

import numpy as np
import scipy.io as sio
import signal
import sys


class UDPRecorder:
    def __init__(self, host='localhost', port=12346):
        self.host = host
        self.port = port
        self.timestamps = []
        self.data_buffer = []
        self.running = False
        self.sock = None

    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\n接收到停止信号，正在保存数据...")
        self.running = False

    def record_udp_data(self, num_floats):
        """录制UDP数据"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.2)  # 设置0.2秒超时，足够响应信号，又不会太频繁
        signal.signal(signal.SIGINT, self.signal_handler)# 注册停止信号处理器
        self.running = True

        print(f"UDP监听在 {self.host}:{self.port}，按Ctrl+C停止...")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(num_floats * 4)
                timestamp = time.time()
            except socket.timeout:
                continue # 如果没收到，直接从while从头开始。这样可以避免ctrl+C杀不掉的问题。

            if len(data) == num_floats * 4:
                floats = struct.unpack(f'{num_floats}f', data)
                self.timestamps.append(timestamp)
                self.data_buffer.append(floats)
                if len(self.timestamps) % 100 == 0 and len(self.timestamps) >= 100:
                    time_diff = timestamp - self.timestamps[-100]
                    frame_rate = 100 / (time_diff+1.e-6)
                    print(f"已录制 {len(self.timestamps)} 帧数据，"
                          f"平均帧率 {frame_rate:.2f} FPS，"
                          f"数据速率 {num_floats * 4 * frame_rate / 1024:.2f} KiB/s")
            else:
                print(f"接收数据异常: 期望收到{num_floats * 4}长度，实际收到{len(data)}")
                break

        #从while出来之后关闭socket。
        self.sock.close()
        print("Socket已关闭")


    def save_to_mat(self):

        filename = f"udp_data_{time.strftime('%Y%m%d_%H%M%S')}.mat"
        timestamps_array = np.array(self.timestamps).reshape(-1, 1)
        data_array = np.array(self.data_buffer)

        # 保存到MAT文件
        mat_data = {
            'timestamps': timestamps_array,
            'udp_data': data_array
        }

        sio.savemat(filename, mat_data)
        print(f"数据已保存到 {filename}")
        print(f"时间戳矩阵形状: {timestamps_array.shape}")
        print(f"UDP数据矩阵形状: {data_array.shape}")




def main():
    # 配置参数
    HOST = 'localhost'  # 监听所有接口
    PORT = 12346  # UDP端口

    # 读取当前配置的复制体数量
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Bennett Foddy\Getting Over It", 0, winreg.KEY_READ)
    num_duplis, _ = winreg.QueryValueEx(key, "numDuplis_h1216609508")
    winreg.CloseKey(key)
    print(f"复制体数量：{num_duplis}")

    # 创建对应数量的录制器
    recorder = UDPRecorder(HOST, PORT)
    recorder.record_udp_data(num_floats=num_duplis * 39)

    # 保存数据
    if recorder.timestamps: recorder.save_to_mat()
    else: print("没有录制到数据")

if __name__ == "__main__":
    main()