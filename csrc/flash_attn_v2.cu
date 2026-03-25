#include <float.h>
#include <stdio.h>

/*
 * Warp-cooperative FlashAttention forward pass (v2)
 *
 * Optimisations over the scalar-per-thread v1 kernel:
 *
 * 1. Warp-level parallelism
 *    32 threads cooperate on each query row, partitioning the head
 *    dimension D across lanes.  Dot products (Q·K^T) are computed as
 *    partial sums and reduced via __shfl_down_sync.
 *
 * 2. Lower register pressure
 *    Each thread holds D/32 elements of Q and O (2 for d=64, 4 for
 *    d=128) versus the full D elements in v1.  The reduction enables
 *    higher occupancy and better latency hiding.
 *
 * 3. Bank-conflict-free shared memory
 *    Thread k owns head-dimension indices {k, k+32, k+64, ...}.
 *    Every warp access to K_tile or V_tile therefore hits bank
 *    (k % 32), guaranteeing zero bank conflicts across the warp.
 *
 * 4. Cooperative tile loading
 *    All warps in the block participate in loading KV tiles into
 *    shared memory, amortising global-memory latency across the
 *    full thread block.
 *
 * Layout note
 * -----------
 * The stride-32 interleaving means that logically contiguous head-
 * dimension elements are NOT held by adjacent threads.  This is
 * intentional: it trades sequential locality (irrelevant in a warp,
 * where all lanes execute simultaneously) for conflict-free bank
 * access, which directly reduces shared-memory stalls.
 */

#define WARP_SIZE 32
#define Bc 32          /* KV tile height — matches warp width */

template <int D, int NUM_WARPS>
__global__ void flash_attn_warp(float *Q, float *K, float *V, float *O,
                                int N) {
    static_assert(D % WARP_SIZE == 0,
                  "Head dimension must be divisible by warp size (32)");

    constexpr int Br  = NUM_WARPS;          /* Q rows per block       */
    constexpr int EPT = D / WARP_SIZE;      /* elements per thread    */

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int q_row   = blockIdx.x * Br + warp_id;
    const bool valid   = (q_row < N);

    /* ── Load this thread's slice of the query row into registers ── */
    float q_reg[EPT];
    float o_reg[EPT];

    if (valid) {
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            q_reg[i] = Q[q_row * D + lane_id + i * WARP_SIZE];
            o_reg[i] = 0.0f;
        }
    }

    /* ── Shared-memory tiles for K and V ──
     *
     * Flat layout, row-major: element (row j, col c) lives at
     * index j*D + c.  With the stride-32 thread mapping, thread k
     * reads col (k + i*32), i.e. index j*D + k + i*32.
     *
     * Bank for that word = (j*D + k + i*32) % 32
     *                    = k % 32   (since D and 32 are both
     *                                multiples of 32)
     *
     * All 32 lanes hit distinct banks → zero conflicts.
     */
    __shared__ float K_tile[Bc * D];
    __shared__ float V_tile[Bc * D];

    const float scale = rsqrtf((float)D);
    float m = -FLT_MAX;                     /* running row-max        */
    float l = 0.0f;                         /* running normaliser     */

    const int total_threads = NUM_WARPS * WARP_SIZE;
    const int tile_elems    = Bc * D;

    /* ── Main loop over KV tiles ── */
    for (int kv_tile = 0; kv_tile < N / Bc; kv_tile++) {
        const int kv_base = kv_tile * Bc;

        /* ── Cooperative tile load ──
         * All threads in the block help fill both tiles.  The flat
         * index directly maps to the global layout because K and V
         * are also stored row-major with stride D.
         */
        for (int idx = threadIdx.x; idx < tile_elems; idx += total_threads) {
            K_tile[idx] = K[kv_base * D + idx];
            V_tile[idx] = V[kv_base * D + idx];
        }
        __syncthreads();

        if (valid) {
            float m_new = m;
            float scores[Bc];

            /* ── Compute Q · K^T scores ──
             * Each thread computes a partial dot product over its
             * EPT elements, then a warp shuffle tree-reduction sums
             * the 32 partials.  Lane 0 broadcasts the result.
             *
             * Cost per KV row: EPT multiply-adds + 5 shuffle steps.
             * Total arithmetic per thread is O(Bc·EPT) — a 32×
             * reduction over v1's O(Bc·D) per thread.
             */
            for (int j = 0; j < Bc; j++) {
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < EPT; i++) {
                    dot += q_reg[i] *
                           K_tile[j * D + lane_id + i * WARP_SIZE];
                }

                /* Warp-wide tree reduction */
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0;
                     offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffff, dot, offset);
                }

                /* Broadcast full dot product to every lane */
                float s  = __shfl_sync(0xffffffff, dot, 0) * scale;
                scores[j] = s;
                m_new = fmaxf(m_new, s);
            }

            /* ── Online softmax rescaling ──
             * Identical to v1: correct the running accumulator and
             * normaliser for the new tile's maximum.  Because the
             * scores were broadcast, m and l are identical across
             * all lanes in the warp — the online-softmax invariant
             * is maintained without extra communication.
             */
            float correction = expf(m - m_new);
            float l_new = l * correction;

            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                o_reg[i] *= correction;
            }

            /* ── Accumulate softmax(S) · V ──
             * Each thread accumulates its own D/32 output slice.
             * No cross-lane reduction is needed here — the scores
             * (and therefore the softmax weights pj) are the same
             * on every lane, and each lane independently owns its
             * share of the output dimension.
             */
            for (int j = 0; j < Bc; j++) {
                float pj = expf(scores[j] - m_new);
                l_new += pj;

                #pragma unroll
                for (int i = 0; i < EPT; i++) {
                    o_reg[i] += pj *
                                V_tile[j * D + lane_id + i * WARP_SIZE];
                }
            }

            m = m_new;
            l = l_new;
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


/* ── Host-side dispatch ── */

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
