// lidar_gen_cuda.cpp
#include <torch/extension.h>

torch::Tensor lidar_gen_cuda_forward(
    torch::Tensor radar_pos,
    torch::Tensor prefab_indices,
    torch::Tensor static_segs,
    torch::Tensor movable_A,
    torch::Tensor movable_B,
    float x1, float y1, float th1,
    float x2, float y2, float th2
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lidar_gen_cuda_forward, "Lidar Data Generation (CUDA)");
}