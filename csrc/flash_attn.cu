#include <float.h>
#include <stdio.h>

#define Br 64
#define Bc 64

template <int D>
__global__ void flash_attn(float *Q, float *K, float *V, float *O, int N) {
    int q_tile_idx = blockIdx.x*Br;
    int tid = threadIdx.x;
    int q_row = q_tile_idx + tid;

    if (q_row >= N) return;


    float q_reg[D];
    float o_reg[D];

    for (int i = 0; i < D; i++) {
        q_reg[i] = Q[q_row*D + i];
        o_reg[i] = 0.0f;
    }

    __shared__ float K_tile[Bc][D];
    __shared__ float V_tile[Bc][D];

    float scale = 1.0f / sqrtf((float) D);
    float m = -FLT_MAX;
    float l = 0.0f;
    
    for (int kv_tile = 0; kv_tile < N / Bc; kv_tile++) {
        int kv_idx = kv_tile*Bc;
        for (int i = 0; i < D; i++) {
            // Note that we are implicitly relying on Br == Bc here. Potential improvement area.
            K_tile[tid][i] = K[(kv_idx + tid)*D + i];
            V_tile[tid][i] = V[(kv_idx + tid)*D + i];
        }

        __syncthreads();

        float s[Bc];
        float m_new = m;
        for (int j = 0; j < Bc; j++) {
            s[j] = 0.0f;
            for (int i = 0; i < D; i++) {
                s[j] += q_reg[i]*K_tile[j][i];
            }
            s[j] *= scale;
            m_new = fmaxf(m_new, s[j]); // Track running max
        }

        float correction = expf(m - m_new);
        float l_new = l*correction;

        float p[Bc];
        for (int j = 0; j < Bc; j++) {
            p[j] = expf(s[j] - m_new);
            l_new += p[j];
        }

        for (int i = 0; i < D; i++) {
            o_reg[i] *= correction;

            for (int j = 0; j < Bc; j++) {
                o_reg[i] += p[j] * V_tile[j][i];
            }
        }

        m = m_new;
        l = l_new;

        __syncthreads();
    }

    for (int i = 0; i < D; i++) {
        O[q_row*D + i] = o_reg[i] / l;
    }

}


void flash_attn_dispatch(float *Q, float *K, float *V, float *O, int N, int d) {
    switch(d) {
        case 64:
            flash_attn<64><<<N/Br, Br>>>(Q, K, V, O, N);
            break;
        case 128:
            flash_attn<128><<<N/Br, Br>>>(Q, K, V, O, N);
            break;
        default:
            fprintf(stderr, "Unsupported head dimension: %d\n", d);
    }
}