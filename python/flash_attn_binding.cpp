#include <torch/extension.h>

void flash_attn_dispatch(float *Q, float *K, float *V, float *O, int N, int d);

torch::Tensor flash_attn_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {

    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    int N = Q.size(0);
    int d = Q.size(1);

    TORCH_CHECK(N % 32 == 0, "N must be divisible by Bc/Br (32)");
    TORCH_CHECK(d == 64 || d == 128, "d must be 64 or 128");

    auto O = torch::zeros({N, d}, Q.options());

    flash_attn_dispatch(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, d
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn", &flash_attn_forward, "FlashAttention forward pass");
}