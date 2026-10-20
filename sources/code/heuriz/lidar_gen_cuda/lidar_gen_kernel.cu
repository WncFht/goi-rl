// lidar_gen_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

constexpr int N_RAYS = 128;
constexpr int N_STATIC = 232;
constexpr int N_MOVABLE_A = 20;
constexpr int N_MOVABLE_B = 4;
constexpr int N_TOTAL_MOVABLE = N_MOVABLE_A + N_MOVABLE_B;
constexpr float MAX_DIST = 20.0f;
constexpr float EPS = 1e-8f;

__constant__ float COS_TABLE[N_RAYS];
__constant__ float SIN_TABLE[N_RAYS];

__device__ __forceinline__ float intersect_ray_segment(
    float rx, float ry,
    float dx, float dy,
    float sx1, float sy1,
    float sx2, float sy2
) {
    float seg_x = sx2 - sx1;
    float seg_y = sy2 - sy1;
    float denom = dx * seg_y - dy * seg_x;
    
    if (fabsf(denom) < EPS) return MAX_DIST;

    float diffx = sx1 - rx;
    float diffy = sy1 - ry;
    float t = (diffx * seg_y - diffy * seg_x) / denom;
    float u = (diffx * dy - diffy * dx) / denom;

    if (t >= 0.0f && t <= MAX_DIST && u >= 0.0f && u <= 1.0f) {
        return t;
    }
    return MAX_DIST;
}

__global__ void lidar_gen_kernel(
    const float* __restrict__ radar_pos, // (M, 2) Tensor
    const int* __restrict__ prefab_indices,
    const float* __restrict__ static_segs,
    const float* __restrict__ movable_A,
    const float* __restrict__ movable_B,
    float x1, float y1, float s1, float c1,
    float x2, float y2, float s2, float c2,
    float* __restrict__ output,
    int M
) {
    // 策略：Block = Lidar, Thread = Ray
    int lidar_idx = blockIdx.x;
    int ray_idx = threadIdx.x;

    if (lidar_idx >= M) return;

    // --- Shared Memory ---
    // 存储变换后的移动物体坐标 (24 * 4 floats)
    __shared__ float4 smem_movable[N_TOTAL_MOVABLE];
    // 存储当前 Grid 的静态物体索引 (232 ints)
    __shared__ int smem_static_indices[N_STATIC];
    // 存储当前 Lidar 位置
    __shared__ float2 smem_lidar_pos;

    // --- Phase 1: 数据加载与预计算 ---
    
    // 1.1 Thread 0 加载并处理 Lidar 位置
    if (ray_idx == 0) {
        // 假设输入是 (M, 2) 连续内存: x0, y0, x1, y1...
        float rx = radar_pos[lidar_idx * 2];
        float ry = radar_pos[lidar_idx * 2 + 1];
        
        rx = fmaxf(-100.0f, fminf(199.999f, rx));
        ry = fmaxf(-100.0f, fminf(599.999f, ry));
        
        smem_lidar_pos = make_float2(rx, ry);
    }
    
    // 1.2 前 24 个线程预计算移动物体坐标 (只需计算一次，所有射线共享)
    if (ray_idx < N_TOTAL_MOVABLE) {
        float lx1, ly1, lx2, ly2;
        float gx1, gy1, gx2, gy2;
        
        if (ray_idx < N_MOVABLE_A) {
            int k = ray_idx;
            lx1 = movable_A[k * 4 + 0]; ly1 = movable_A[k * 4 + 1];
            lx2 = movable_A[k * 4 + 2]; ly2 = movable_A[k * 4 + 3];
            
            gx1 = lx1 * c1 - ly1 * s1 + x1;
            gy1 = lx1 * s1 + ly1 * c1 + y1;
            gx2 = lx2 * c1 - ly2 * s1 + x1;
            gy2 = lx2 * s1 + ly2 * c1 + y1;
        } else {
            int k = ray_idx - N_MOVABLE_A;
            lx1 = movable_B[k * 4 + 0]; ly1 = movable_B[k * 4 + 1];
            lx2 = movable_B[k * 4 + 2]; ly2 = movable_B[k * 4 + 3];
            
            gx1 = lx1 * c2 - ly1 * s2 + x2;
            gy1 = lx1 * s2 + ly1 * c2 + y2;
            gx2 = lx2 * c2 - ly2 * s2 + x2;
            gy2 = lx2 * s2 + ly2 * c2 + y2;
        }
        smem_movable[ray_idx] = make_float4(gx1, gy1, gx2, gy2);
    }

    __syncthreads(); // 确保位置和移动物体数据准备就绪

    // 1.3 协作加载静态索引
    float rx = smem_lidar_pos.x;
    float ry = smem_lidar_pos.y;
    int gx = min((int)((rx + 100.0f) * 0.5f), 149);
    int gy = min((int)((ry + 100.0f) * 0.5f), 349);
    int grid_base = (gy * 150 + gx) * N_STATIC;

    // 128个线程协作加载232个整数
    for (int i = ray_idx; i < N_STATIC; i += blockDim.x) {
        smem_static_indices[i] = prefab_indices[grid_base + i];
    }
    
    __syncthreads(); // 确保索引加载完毕

    // --- Phase 2: 并行射线计算 ---
    
    // 每个线程处理一条射线
    if (ray_idx < N_RAYS) {
        float dir_x = COS_TABLE[ray_idx];
        float dir_y = SIN_TABLE[ray_idx];
        float min_t = MAX_DIST;

        // 2.1 检测移动物体 (全在 Shared Memory，极快)
        #pragma unroll
        for (int k = 0; k < N_TOTAL_MOVABLE; ++k) {
            float4 seg = smem_movable[k];
            float t = intersect_ray_segment(rx, ry, dir_x, dir_y, seg.x, seg.y, seg.z, seg.w);
            min_t = fminf(min_t, t);
        }

        // 2.2 检测静态物体 (索引在 Shared，几何在 Global)
        for (int k = 0; k < N_STATIC; ++k) {
            int seg_id = smem_static_indices[k];
            
            // 使用 __ldg 提示编译器使用只读缓存 (Texture Cache)
            const float* seg_ptr = &static_segs[seg_id * 4];
            float sx1 = __ldg(&seg_ptr[0]);
            float sy1 = __ldg(&seg_ptr[1]);
            float sx2 = __ldg(&seg_ptr[2]);
            float sy2 = __ldg(&seg_ptr[3]);

            float t = intersect_ray_segment(rx, ry, dir_x, dir_y, sx1, sy1, sx2, sy2);
            min_t = fminf(min_t, t);
        }

        output[lidar_idx * N_RAYS + ray_idx] = min_t;
    }
}

void load_angle_tables() {
    static bool loaded = false;
    if (loaded) return;
    float h_cos[N_RAYS], h_sin[N_RAYS];
    for (int i = 0; i < N_RAYS; ++i) {
        float angle = i * (2.0f * M_PI / N_RAYS);
        h_cos[i] = cosf(angle);
        h_sin[i] = sinf(angle);
    }
    cudaMemcpyToSymbol(COS_TABLE, h_cos, N_RAYS * sizeof(float));
    cudaMemcpyToSymbol(SIN_TABLE, h_sin, N_RAYS * sizeof(float));
    loaded = true;
}

torch::Tensor lidar_gen_cuda_forward(
    torch::Tensor radar_pos,
    torch::Tensor prefab_indices,
    torch::Tensor static_segs,
    torch::Tensor movable_A,
    torch::Tensor movable_B,
    float x1, float y1, float th1,
    float x2, float y2, float th2
) {
    load_angle_tables();
    int M = radar_pos.size(0);
    auto output = torch::empty({M, N_RAYS}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 block(N_RAYS); // 128
    dim3 grid(M);       // M

    float s1 = sinf(th1), c1 = cosf(th1);
    float s2 = sinf(th2), c2 = cosf(th2);

    lidar_gen_kernel<<<grid, block>>>(
        radar_pos.data_ptr<float>(),
        prefab_indices.data_ptr<int>(),
        static_segs.data_ptr<float>(),
        movable_A.data_ptr<float>(),
        movable_B.data_ptr<float>(),
        x1, y1, s1, c1,
        x2, y2, s2, c2,
        output.data_ptr<float>(),
        M
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }
    return output;
}