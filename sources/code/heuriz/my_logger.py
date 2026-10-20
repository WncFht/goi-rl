from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import threading
from queue import Queue
import time

class Logger:
    def __init__(self):
        self.logger=SummaryWriter(f'data/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.queue = Queue()
        self.thread = threading.Thread(target=self._write_worker, daemon=True)
        self.thread.start()


    def _write_worker(self):
        while True:
            data,step = self.queue.get()
            for k, v in data.items():
                self.logger.add_scalar(k, v, step)
            self.queue.task_done()

    def log_dict_collect(self, data: dict[str, float],step):
        self.queue.put((data,step))


    def close(self):
        self.queue.join()  # 等待所有任务完成
        self.logger.close()