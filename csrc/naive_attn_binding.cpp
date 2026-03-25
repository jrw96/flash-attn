#include <torch/extension.h>

void naive_attn(float *Q, float *K, float *V, float *C, float *O, int N, int d);

torch::Tensor naive_attn_forward(
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

    TORCH_CHECK(d == 64 || d == 128, "d must be 64 or 128");

    auto C = torch::zeros({N, N}, Q.options()); // Intermediate storage matrix
    auto O = torch::zeros({N, d}, Q.options());

    naive_attn(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        C.data_ptr<float>(),
        O.data_ptr<float>(),
        N, d
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_attn", &naive_attn_forward, "Naive attention forward pass");
}