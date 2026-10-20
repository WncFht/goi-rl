import torch
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


# 批量矩形与线段求交测试
# 矩形 (M, 4) tensor, 格式 (left, bottom, right, top)
# 线段 (N, 4) tensor, 格式 (x1, y1, x2, y2)
# 返回 (M, N) bool tensor，True 表示相交
def rects_segments_intersect(rects, segments) -> torch.Tensor:
    x1 = segments[:, 0].unsqueeze(0)  # (1, N)
    y1 = segments[:, 1].unsqueeze(0)
    x2 = segments[:, 2].unsqueeze(0)
    y2 = segments[:, 3].unsqueeze(0)
    left = rects[:, 0].unsqueeze(1)  # (M, 1)
    bottom = rects[:, 1].unsqueeze(1)
    right = rects[:, 2].unsqueeze(1)
    top = rects[:, 3].unsqueeze(1)
    dx = x2 - x1  # broadcast (1, N)
    dy = y2 - y1

    # Start with all true. For each slab, compute t_min and t_max. Use Liang-Barsky algorithm in vectorized form
    intersect = torch.ones(left.shape[0], x1.shape[1], dtype=torch.bool, device=rects.device)
    t0 = torch.zeros_like(intersect, dtype=torch.float32)
    t1 = torch.ones_like(intersect, dtype=torch.float32)

    # Helper: update t0/t1 for one slab. p: (M, N), q: (M, N)
    def update_slab(p, q, t0, t1, intersect):
        eps = 1e-8
        parallel = torch.abs(p) < eps  # If parallel and outside slab -> no intersection
        no_intersect = parallel & (q < 0)
        r = q / (p + eps)  # For non-parallel
        t0_new = torch.where(p < 0, torch.maximum(t0, r), t0)
        t1_new = torch.where(p > 0, torch.minimum(t1, r), t1)
        t0 = torch.where(~parallel, t0_new, t0)
        t1 = torch.where(~parallel, t1_new, t1)
        intersect = intersect & (~no_intersect) & (t0 <= t1)
        return t0, t1, intersect

    t0, t1, intersect = update_slab(-dx, x1 - left, t0, t1, intersect)  # Left:-dx*t<=x1-left=>p=-dx,q=x1-left
    t0, t1, intersect = update_slab(dx, right - x1, t0, t1, intersect)  # Right:dx*t<=right-x1=>p=dx,q=right-x1
    t0, t1, intersect = update_slab(-dy, y1 - bottom, t0, t1, intersect)  # Bottom:-dy*t<=y1-bottom
    t0, t1, intersect = update_slab(dy, top - y1, t0, t1, intersect)  # Top:dy*t<=top-y1

    return intersect


# 批量圆与线段求交测试
# 圆 (M, 3) tensor, 格式 (cx, cy, r)
# 线段 (N, 4) tensor, 格式 (x1, y1, x2, y2)
# 返回 (M, N) bool tensor，True 表示相交
def circles_segments_intersect(circles, segments):
    eps = 1e-8
    cx = circles[:, 0].unsqueeze(1)  # (M, 1)
    cy = circles[:, 1].unsqueeze(1)  # (M, 1)
    r = circles[:, 2].unsqueeze(1)  # (M, 1)
    x1 = segments[:, 0].unsqueeze(0)  # (1, N)
    y1 = segments[:, 1].unsqueeze(0)  # (1, N)
    x2 = segments[:, 2].unsqueeze(0)  # (1, N)
    y2 = segments[:, 3].unsqueeze(0)  # (1, N)
    dx = x2 - x1  # (1, N)
    dy = y2 - y1  # (1, N)
    cx_minus_x1 = cx - x1  # (M, N)
    cy_minus_y1 = cy - y1  # (M, N)
    # 投影参数 t = dot(AC, AB) / dot(AB, AB)
    dot_product = cx_minus_x1 * dx + cy_minus_y1 * dy  # (M, N)
    len_sq = dx * dx + dy * dy  # (1, N)
    t = torch.where(len_sq < eps, torch.zeros_like(dot_product), dot_product / len_sq.clamp(min=eps))
    t_clamped = t.clamp(0.0, 1.0)  # (M, N)
    closest_x = x1 + t_clamped * dx  # (M, N)
    closest_y = y1 + t_clamped * dy  # (M, N)
    dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2  # (M, N)
    intersect = dist_sq <= (r * r + eps)  # (M, N)
    return intersect


# return (M, 4) tensor, format (left, bottom, right, top)
def gen_rects(y_min, y_max, y_step, x_min, x_max, x_step, y_margin, x_margin, device) -> torch.Tensor:
    bottoms, lefts = torch.meshgrid(torch.arange(y_min, y_max, y_step, device=device),
                                    torch.arange(x_min, x_max, x_step, device=device), indexing='ij')

    lefts = lefts.reshape(-1)
    bottoms = bottoms.reshape(-1)
    rects = torch.stack([lefts - x_margin,
                         bottoms - y_margin,
                         lefts + x_step + x_margin,
                         bottoms + y_step + y_margin], dim=1)  # Stack into (M, 4) tensor
    return rects


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    grid_step = 2
    rects = gen_rects(-100, 600, grid_step, -100, 200, grid_step, 0, 0, device=device)  # M * (l b r t)

    initial_radius = 512  # 这应该很夸张了，500多米的半径。
    desired_num_segs = 232  # 每个探测区域想要多少条线段
    circles = torch.stack([
        (rects[:, 0] + rects[:, 2]) / 2,
        (rects[:, 1] + rects[:, 3]) / 2,
        torch.zeros(rects.shape[0], dtype=torch.float, device=device)  # 初始半径设成0，方便迭代
    ], dim=1)

    segments = torch.from_numpy(np.loadtxt('all_lines.csv', delimiter=',').astype(np.float32)).to(device)  # N,4
    seg_props = torch.from_numpy(np.loadtxt('all_lines_props.csv', delimiter=',').astype(np.float32)).to(device)  # 24,4

    radius = circles[:, 2]  # 圆的半径单拿出来准备进行负反馈。
    segs: torch.Tensor = torch.zeros_like(radius)  # 长度为区块数量。指的是在对应半径下有多少段线段相交，目标是256。
    result: torch.Tensor = torch.zeros_like(radius)  # 应该为区块数量*线段数量，不过这里无所谓了只是为了不报错。

    for iters in range(30):
        if iters < 29:
            radius -= ((segs > desired_num_segs).float() - 0.5) * initial_radius  # 二分法负反馈，条数多的就缩小范围，少的就放大范围。
            initial_radius /= 2
        else: radius -= (segs > desired_num_segs).float() * 0.0001  # 最后一次迭代来一波大的，让那些还大于desired_num_segs的都收敛一些，其他的保持不变

        result = circles_segments_intersect(circles, segments)  # (M,N) #圆形测试，M个区块，和N个线段是否相交。
        segs = result.sum(dim=1).float()  # M, 和每个圆相交的线段个数。

        max_segs, min_segs, mean_segs, std_segs = segs.max().item(), segs.min().item(), segs.mean().item(), segs.std().item()
        max_radius, min_radius, mean_radius, std_radius = radius.max().item(), radius.min().item(), radius.mean().item(), radius.std().item()
        print(f"iters{iters}: {max_segs=},{min_segs=},{mean_segs=},{std_segs=} and "
              f"{max_radius=},{min_radius=},{mean_radius=},{std_radius=}")

    _, indices = torch.sort(result, dim=1, descending=True)  # 这样为True的index就排在前面了。
    indices = indices[:, :desired_num_segs]  # shape: (M, desired_num_segs)
    print("前五行下标:", indices.cpu().numpy()[:5, :])
    print(indices.shape, segments.shape, seg_props.shape)  # (M, desired_num_segs), (N,4), (24,4)
    torch.save({
        'indices': indices.contiguous().view(700 // grid_step, 300 // grid_step, desired_num_segs),  # 350,150,232
        'segments': segments,  # (4550,4)
        'seg_hat_local': seg_props[:20, :],  # (20,4)
        'seg_bucket_local': seg_props[20:, :]  # (4,4)
    }, "related_seg_idxs.pt")
    print("Done.")


if __name__ == "__main__":
    main()
