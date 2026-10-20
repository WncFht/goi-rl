import threading
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

#顺序全部同步成：ob,ac,nex_ob,rew,done

class ThreadSafeReplayBuffer:
    def __init__(self, ob_dim: int, ac_dim: int, capacity=1048576): #每秒能产生64*120=7680，我可以让缓冲区支持大概两分多钟。
        self.ob:ndarray = np.zeros((capacity,ob_dim),dtype=np.float32)
        self.ac:ndarray = np.zeros((capacity,ac_dim),dtype=np.float32)
        self.nex_ob:ndarray = np.zeros((capacity,ob_dim),dtype=np.float32)
        self.rew:ndarray = np.zeros(capacity, dtype=np.float32)
        self.done:ndarray = np.zeros(capacity, dtype=np.uint8)

        self.capacity=capacity
        self.write_ptr = 0  # next index to write
        self.current_size = 0  # number of valid samples in buffer

        # 线程安全锁
        self.write_lock = threading.RLock()

    # 这个done的处理是这样的，truncated不算done，只有die了或者过关了才算done。不过truncated的那一帧其实是需要丢掉的，需要把上一帧变成done。
    def push(self, ob: ndarray, ac: ndarray, nex_ob: ndarray, rew: ndarray, done: ndarray) -> None:
        batch_size=ob.shape[0]
        assert batch_size == ac.shape[0] == nex_ob.shape[0] == rew.shape[0] == done.shape[0]
        with self.write_lock:
            write_to = self.write_ptr + batch_size
            if write_to <= self.capacity:  # 没有跨越循环边界
                self.ob[self.write_ptr:write_to] = ob
                self.ac[self.write_ptr:write_to] = ac
                self.nex_ob[self.write_ptr:write_to] = nex_ob
                self.rew[self.write_ptr:write_to] = rew
                self.done[self.write_ptr:write_to] = done
                self.current_size = max(write_to, self.current_size)  # write_to走到过的最远的地方就是当前大小，最大等于capacity。
                self.write_ptr = write_to % self.capacity  # 写指针直接更新到下一个开始写的地方，正好收尾就回0。
            else:  # 跨越了循环边界：先写前一半，再写后一半。
                first_part_len = self.capacity - self.write_ptr
                next_part_len = write_to - self.capacity
                self.ob[self.write_ptr:] = ob[:first_part_len]
                self.ac[self.write_ptr:] = ac[:first_part_len]
                self.nex_ob[self.write_ptr:] = nex_ob[:first_part_len]
                self.rew[self.write_ptr:] = rew[:first_part_len]
                self.done[self.write_ptr:] = done[:first_part_len]
                self.ob[:next_part_len] = ob[first_part_len:]
                self.ac[:next_part_len] = ac[first_part_len:]
                self.nex_ob[:next_part_len] = nex_ob[first_part_len:]
                self.rew[:next_part_len] = rew[first_part_len:]
                self.done[:next_part_len] = done[first_part_len:]
                self.current_size = self.capacity  # 发生过跨越说明已经满了。
                self.write_ptr = next_part_len



    #采样的顺序：ob,ac,nex_ob,rew,done
    def sample(self,read_batchsize:int=512)-> tuple[ndarray,ndarray,ndarray,ndarray,ndarray]:
        with self.write_lock:
            assert self.current_size >= read_batchsize
            idx = np.random.randint(0, self.current_size, size=read_batchsize)
            return np.copy(self.ob[idx]), np.copy(self.ac[idx]),np.copy(self.nex_ob[idx]), np.copy(self.rew[idx]),np.copy(self.done[idx])



class ThreadSafeReplayBufferTorch:
    def __init__(self, ob_dim: int, ac_dim: int, capacity=1048576, device:torch.device='cuda'): #每秒能产生64*120=7680，我可以让缓冲区支持大概五分钟。
        print("创建torch缓冲区中...")
        # assert capacity % write_batchsize == 0, "capacity must be divisible by write_batchsize" #现在不必了。
        self.device=device

        self.ob:Tensor = torch.zeros((capacity,ob_dim),dtype=torch.float32, device=device)
        self.ac:Tensor = torch.zeros((capacity,ac_dim),dtype=torch.float32, device=device)
        self.nex_ob:Tensor = torch.zeros((capacity,ob_dim),dtype=torch.float32, device=device)
        self.rew:Tensor = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.done:Tensor = torch.zeros(capacity, dtype=torch.uint8, device=device)

        self.capacity=capacity
        self.write_ptr = 0  # next index to write
        self.current_size = 0  # number of valid samples in buffer
        print(f"torch缓冲区创建完成! 容量{capacity}，设备{device}")
        # Locking
        self.write_lock = threading.RLock()

    # 写入一批，注意batchsize已经固定了，这样可以避免一批之内跨首尾的问题。
    # 这个done的处理是这样的，truncated不算done，只有die了或者过关了才算done。不算done所以也就不存在那个不知道什么时候done了的问题了。
    # def push(self, ob: ndarray, ac: ndarray, nex_ob: ndarray, rew: ndarray, done: ndarray) -> None:
    #     batch_size = ob.shape[0]
    #     assert batch_size == ac.shape[0] == nex_ob.shape[0] == rew.shape[0] == done.shape[0]
    #     ob_t = torch.as_tensor(ob, device=self.device)
    #     ac_t = torch.as_tensor(ac, device=self.device)
    #     nex_ob_t = torch.as_tensor(nex_ob, device=self.device)
    #     rew_t = torch.as_tensor(rew, device=self.device)
    #     done_t = torch.as_tensor(done, device=self.device)
    def push(self, ob_t: Tensor, ac_t: Tensor, nex_ob_t: Tensor, rew_t: Tensor, done_t: Tensor) -> None:
        batch_size = ob_t.shape[0]
        assert batch_size == ac_t.shape[0] == nex_ob_t.shape[0] == rew_t.shape[0] == done_t.shape[0]
        with self.write_lock:
            write_to = self.write_ptr + batch_size
            if write_to <= self.capacity:  # 没有跨越循环边界
                self.ob[self.write_ptr:write_to] = ob_t
                self.ac[self.write_ptr:write_to] = ac_t
                self.nex_ob[self.write_ptr:write_to] = nex_ob_t
                self.rew[self.write_ptr:write_to] = rew_t
                self.done[self.write_ptr:write_to] = done_t
                self.current_size = max(write_to, self.current_size)  # write_to走到过的最远的地方就是当前大小，最大等于capacity。
                self.write_ptr = write_to % self.capacity  # 写指针直接更新到下一个开始写的地方，正好收尾就回0。
            else:  # 跨越了循环边界：先写前一半，再写后一半。
                first_part_len = self.capacity - self.write_ptr
                next_part_len = write_to - self.capacity
                self.ob[self.write_ptr:] = ob_t[:first_part_len]
                self.ac[self.write_ptr:] = ac_t[:first_part_len]
                self.nex_ob[self.write_ptr:] = nex_ob_t[:first_part_len]
                self.rew[self.write_ptr:] = rew_t[:first_part_len]
                self.done[self.write_ptr:] = done_t[:first_part_len]
                self.ob[:next_part_len] = ob_t[first_part_len:]
                self.ac[:next_part_len] = ac_t[first_part_len:]
                self.nex_ob[:next_part_len] = nex_ob_t[first_part_len:]
                self.rew[:next_part_len] = rew_t[first_part_len:]
                self.done[:next_part_len] = done_t[first_part_len:]
                self.current_size = self.capacity  # 发生过跨越说明已经满了。
                self.write_ptr = next_part_len

    #采样的顺序：ob,ac,nex_ob,rew,done.
    def sample(self,read_batchsize:int=512)-> tuple[Tensor,Tensor,Tensor,Tensor,Tensor]:
        with self.write_lock:
            assert self.current_size >= read_batchsize
            idx = torch.randint(0, self.current_size, (read_batchsize,), device=self.device)
            return self.ob[idx].clone(), self.ac[idx].clone(), self.nex_ob[idx].clone(), self.rew[idx].clone(), self.done[idx].clone()
