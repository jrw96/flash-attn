#include <float.h>
#include <stdio.h>

/*
 * Warp-cooperative FlashAttention forward pass (v2)
 *
 * Each warp owns one query row.  32 lanes partition the head dimension
 * D, computing dot products via partial sums + warp shuffle reduction.
 * All warps in the block share KV tiles in shared memory.
 *
 * Key optimisation over the tiled-scores variant
 * -----------------------------------------------
 * The online softmax update is applied per KV row rather than per tile.
 * This eliminates the float scores[Bc] register array (32 regs/thread),
 * roughly halving per-thread register pressure:
 *
 *   With scores[32]: ~50 regs/thread x 1024 threads = 51200 regs/block
 *     -> 1 block/SM on T4 (65536 regs/SM)
 *
 *   Without scores:  ~20 regs/thread x 1024 threads = 20480 regs/block
 *     -> 2 blocks/SM on T4 (thread-limited, not register-limited)
 *
 * The tradeoff is more exp() evaluations (one correction per KV row
 * instead of one per tile), but this is offset by doubled occupancy
 * and better latency hiding from concurrent blocks.
 *
 * Shared memory layout
 * --------------------
 * Flat row-major: element (row j, col c) at index j*D + c.
 * Thread k accesses cols {k, k+32, k+64, ...}, hitting bank k%32.
 * All 32 lanes hit distinct banks -> zero bank conflicts.
 */

#define WARP_SIZE 32
#define Bc 32

template <int D, int NUM_WARPS>
__global__ void flash_attn_warp(float *Q, float *K, float *V, float *O,
                                int N) {
    static_assert(D % WARP_SIZE == 0,
                  "Head dimension must be divisible by warp size (32)");

    constexpr int Br  = NUM_WARPS;
    constexpr int EPT = D / WARP_SIZE;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int q_row   = blockIdx.x * Br + warp_id;
    const bool valid  = (q_row < N);

    /* ── Load query row (each lane holds D/32 elements) ── */
    float q_reg[EPT];
    float o_reg[EPT];

    if (valid) {
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            q_reg[i] = Q[q_row * D + lane_id + i * WARP_SIZE];
            o_reg[i] = 0.0f;
        }
    }

    __shared__ float K_tile[Bc * D];
    __shared__ float V_tile[Bc * D];

    const float scale = rsqrtf((float)D);
    float m = -FLT_MAX;
    float l = 0.0f;

    const int total_threads = NUM_WARPS * WARP_SIZE;
    const int tile_elems    = Bc * D;

    for (int kv_tile = 0; kv_tile < N / Bc; kv_tile++) {
        const int kv_base = kv_tile * Bc;

        /* Cooperative tile load */
        for (int idx = threadIdx.x; idx < tile_elems; idx += total_threads) {
            K_tile[idx] = K[kv_base * D + idx];
            V_tile[idx] = V[kv_base * D + idx];
        }
        __syncthreads();

        if (valid) {
            /* ── Process each KV row individually ──
             *
             * Instead of computing all Bc scores, storing them, then
             * accumulating, we compute one score at a time and
             * immediately apply the online softmax update.
             *
             * This eliminates the scores[Bc] register array at the
             * cost of evaluating the correction factor per row
             * rather than per tile.  When m doesn't change (the
             * common case after the first few rows), correction = 1.0
             * and the compiler can often elide the multiply.
             */
            for (int j = 0; j < Bc; j++) {
                /* ── Warp-parallel dot product: Q_row . K_tile[j] ── */
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < EPT; i++) {
                    dot += q_reg[i] *
                           K_tile[j * D + lane_id + i * WARP_SIZE];
                }

                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0;
                     offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffff, dot, offset);
                }

                float s = __shfl_sync(0xffffffff, dot, 0) * scale;

                /* ── Per-row online softmax update ── */
                float m_new = fmaxf(m, s);
                float correction = expf(m - m_new);
                float pj = expf(s - m_new);

                /* Rescale running accumulator and normaliser */
                l = l * correction + pj;

                #pragma unroll
                for (int i = 0; i < EPT; i++) {
                    o_reg[i] = o_reg[i] * correction +
                               pj * V_tile[j * D + lane_id + i * WARP_SIZE];
                }

                m = m_new;
            }
        }
        __syncthreads();
    }

    /* ── Write normalised output ── */
    if (valid) {
        float inv_l = 1.0f / l;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            O[q_row * D + lane_id + i * WARP_SIZE] = o_reg[i] * inv_l;
        }
    }
}


void flash_attn_v2_dispatch(float *Q, float *K, float *V, float *O,
                            int N, int d) {
    constexpr int NUM_WARPS = 32;
    constexpr int Br = NUM_WARPS;
    const int grid  = (N + Br - 1) / Br;
    const int block = NUM_WARPS * WARP_SIZE;

    switch (d) {
        case 64:
            flash_attn_warp<64,  NUM_WARPS>
                <<<grid, block>>>(Q, K, V, O, N);
            break;
        case 128:
            flash_attn_warp<128, NUM_WARPS>
                <<<grid, block>>>(Q, K, V, O, N);
            break;
        default:
            fprintf(stderr, "Unsupported head dimension: %d\n", d);
    }
}
