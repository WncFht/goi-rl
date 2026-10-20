import torch
import lidar_gen_cuda
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)



class UltraFastLidarRenderer:
    def __init__(self, prefab_path, device='cuda'):
        assert device == 'cuda', "CUDA only"
        data = torch.load(prefab_path, weights_only=True, map_location='cuda')
        self.indices = data['indices'].contiguous().to(torch.int32)  # (350,150,232)
        self.segments = data['segments'].contiguous()  # (S,4)
        self.segA = data['seg_hat_local'].contiguous()  # (20,4)
        self.segB = data['seg_bucket_local'].contiguous()  # (4,4)

    @torch.no_grad()
    def __call__(self, lidar_pos: torch.Tensor, pose1, pose2):
        assert lidar_pos.is_cuda and lidar_pos.dtype == torch.float32
        x1, y1, th1 = pose1
        x2, y2, th2 = pose2
        return lidar_gen_cuda.forward(
            lidar_pos,
            self.indices,
            self.segments,
            self.segA,
            self.segB,
            float(x1), float(y1), float(th1*0.017453292519943295),
            float(x2), float(y2), float(th2*0.017453292519943295)
        )


def benchmark():
    gen = UltraFastLidarRenderer("assets/related_seg_idxs.pt")

    pose1 = (61.21, 242.16, 0)  # snowhat (x, y, theta)
    pose2 = (17.23, 270.17, 0)  # bucket (x, y, theta)

    print("[INFO] Warming up...")

    for _ in range(10):
        lidar_pos = torch.zeros(32, 2, device='cuda')
        gen(lidar_pos, pose1, pose2)

    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        # 模拟雷达位置：在场景范围内随机生成
        torch.manual_seed(42)
        lidar_pos = torch.zeros(M, 2, device='cuda')

        print("[INFO] Running benchmark...")
        num_iters = 100
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        all_random_sum = 0
        all_ray_sum=0

        # Record timing
        start_event.record()
        for _ in range(num_iters):
            lidar_pos[:, 0].uniform_(-100, 200)  # x in [-100, 200]
            lidar_pos[:, 1].uniform_(-100, 600)  # y in [-100, 600]
            all_random_sum+=lidar_pos.sum().item()
            dist = gen(lidar_pos, pose1, pose2)
            all_ray_sum+=dist.sum().item()

        end_event.record()

        torch.cuda.synchronize()  # Wait for GPU to finish
        elapsed_ms = start_event.elapsed_time(end_event)
        avg_us = (elapsed_ms * 1000) / num_iters
        avg_pos=all_random_sum/M/num_iters
        avg_ray_len=all_ray_sum/M/128/num_iters

        # Optional: sanity check
        assert dist.shape == (M, 128)
        assert dist.dtype == torch.float32
        assert torch.all((dist >= 0) & (dist <= 20.0 + 1e-5))

        print(f"\n✅ Benchmark Result (num lidars={M}:")
        print(f"  Average latency: {avg_us:.2f} μs")
        print(f"  Throughput:      {1e6 / avg_us:.1f} FPS; {1e6 * M / avg_us:.1f} LidarPS")
        print(f"  Average pos:{avg_pos}, Average ray len:{avg_ray_len}")


if __name__ == "__main__":
    gen = UltraFastLidarRenderer("assets/related_seg_idxs.pt")

    pose1 = (61.21, 242.16, 45)  # snowhat (x, y, theta)
    pose2 = (17.23, 270.17, 0)  # bucket (x, y, theta)
    lidar_pos = torch.tensor([[25.5,72.5]], device='cuda')
    dist = gen(lidar_pos, pose1, pose2)

    print(dist.squeeze().cpu().numpy())
