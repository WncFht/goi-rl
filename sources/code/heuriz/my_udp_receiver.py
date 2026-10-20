import struct
import time
import signal
import socket
import threading

class UDPReceiver:
    def __init__(self, num_floats):
        self.running: bool = True
        self.num_floats = num_floats
        self.num_bytes = num_floats * 4
        self.last100_timestamp = time.perf_counter()
        self.frame: list | None = None
        self.timestamp: float = 0
        self.lock = threading.RLock()
        signal.signal(signal.SIGINT, self.signal_handler)  # 注册停止信号处理器

    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\n接收到停止信号，停止记录UDP")
        self.running = False

    def record_udp_data(self, host, port):
        """录制UDP数据"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.settimeout(0.2)  # 设置0.2秒超时，足够响应信号，又不会太频繁
        print(f"UDP监听在 {host}:{port}，按Ctrl+C停止...")

        frame_cnt = 0
        while self.running:
            try:
                data, addr = sock.recvfrom(self.num_bytes)
                timestamp = time.perf_counter()
            except socket.timeout:
                continue  # 如果没收到，直接从while从头开始。这样可以避免ctrl+C杀不掉的问题。其他问题比如数据格式不对等等直接报错。

            if len(data) == self.num_bytes:
                floats = struct.unpack(f'{self.num_floats}f', data)
                with self.lock:
                    self.frame = floats  # 关键赋值！如果活得不了锁就只能等在这。
                    self.timestamp = timestamp
                    frame_cnt += 1

                if frame_cnt % 100 == 0:
                    frame_rate = 100 / (timestamp - self.last100_timestamp + 1.e-6)
                    self.last100_timestamp = timestamp
                    # print(f"共收到 {frame_cnt} 帧数据，数据帧率 {frame_rate:.2f} FPS")

            else:
                print(f"接收数据异常: 期望收到{self.num_bytes}字节长度，实际收到{len(data)}")
                break

        # 从while出来之后关闭socket。
        sock.close()
        print("Socket已关闭")
