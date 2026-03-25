#include <float.h>
#include <stdio.h>

#include "cublas_v2.h"


__global__ void softmax(float *M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];


    // First, find row-max
    float local_max = -FLT_MAX;
    for(int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, M[row*N + i]);
    }

    smem[tid] = local_max;
    __syncthreads();


    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = smem[0];


    // Next, calculate element-wise exponents and keep track of sum
    float local_sum = 0.0f;
    for(int i = tid; i < N; i += blockDim.x) {
        M[row*N + i] = expf(M[row*N + i] - row_max);
        local_sum += M[row*N + i];
    }

    smem[tid] = local_sum;
    __syncthreads();


    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = smem[0];


    // Finally, normalize by the row sum
    for(int i = tid; i < N; i += blockDim.x) {
        M[row*N + i] /= row_sum;
    }


}


void naive_attn(float *Q, float *K, float *V, float *C, float *O, int N, int d) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f/sqrtf((float) d);
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &alpha, K, d, Q, d, &beta, C, N);
    softmax<<<N, 256, 256*sizeof(float)>>>(C, N);

    alpha = 1.0f;
    beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, &alpha, V, d, C, N, &beta, O, d);

    cublasDestroy(handle);
}