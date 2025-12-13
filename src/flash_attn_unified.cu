#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Unified-memory variant of the FlashAttention-style kernel.

namespace {
constexpr int kBc = 32; // key tile width, assumed <= warp size
constexpr int kBr = 4;  // rows per block (one warp per row)

template<int Bc, int Br>
__global__ void flash_attn_online_um(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     const float* __restrict__ V,
                                     float* __restrict__ O,
                                     int N, int d, int nh, float scale) {
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int row_base = blockIdx.z * Br;
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;
    if (warp_id >= Br) return;
    int row = row_base + warp_id;
    if (row >= N) return;

    extern __shared__ float smem[];
    float* q_rows  = smem;                    // Br * d
    float* o_rows  = q_rows + Br * d;         // Br * d
    float* k_tile  = o_rows + Br * d;         // Bc * d
    float* v_tile  = k_tile + Bc * d;         // Bc * d

    float* q_row  = q_rows  + warp_id * d;
    float* o_row  = o_rows  + warp_id * d;

    for (int i = lane; i < d; i += blockDim.x) {
        q_row[i] = Q[(batch * nh * N * d) + (head * N * d) + (row * d) + i];
        o_row[i] = 0.f;
    }
    __shared__ float m_row[Br];
    __shared__ float l_row[Br];
    if (lane == 0) {
        m_row[warp_id] = -INFINITY;
        l_row[warp_id] = 0.f;
    }
    __syncthreads();

    for (int col_start = 0; col_start < N; col_start += Bc) {
        int k_col = col_start + lane;

        // Load K/V tile cooperatively.
        int t = warp_id * blockDim.x + lane;
        int tile_elems = Bc * d;
        for (int idx = t; idx < tile_elems; idx += blockDim.x * blockDim.y) {
            int col = idx / d;
            int dim = idx - col * d;
            int g_col = col_start + col;
            if (g_col < N) {
                k_tile[idx] = K[(batch * nh * N * d) + (head * N * d) + (g_col * d) + dim];
                v_tile[idx] = V[(batch * nh * N * d) + (head * N * d) + (g_col * d) + dim];
            } else {
                k_tile[idx] = 0.f;
                v_tile[idx] = 0.f;
            }
        }
        __syncthreads();

        float score = -INFINITY;
        if (k_col < N) {
            const float* k_ptr = &k_tile[(k_col - col_start) * d];
            float qk = 0.f;
            for (int i = 0; i < d; ++i) qk += q_row[i] * k_ptr[i];
            score = qk * scale;
        }

        float max_val = score;
        for (int offset = Bc / 2; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        float m_tile = __shfl_sync(0xffffffff, max_val, 0);

        float p = (k_col < N) ? expf(score - m_tile) : 0.f;
        float sum_val = p;
        for (int offset = Bc / 2; offset > 0; offset >>= 1) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        float l_tile = __shfl_sync(0xffffffff, sum_val, 0);

        float m_old = m_row[warp_id];
        float l_old = l_row[warp_id];
        float m_new = fmaxf(m_old, m_tile);
        float l_new = l_old * expf(m_old - m_new) + l_tile * expf(m_tile - m_new);
        float alpha = expf(m_old - m_new) * l_old;
        float beta  = expf(m_tile - m_new);

        for (int i = lane; i < d; i += blockDim.x) {
            float pv = (k_col < N) ? p * v_tile[(k_col - col_start) * d + i] : 0.f;
            for (int offset = Bc / 2; offset > 0; offset >>= 1) {
                pv += __shfl_down_sync(0xffffffff, pv, offset);
            }
            float o_new = (o_row[i] * alpha + pv * beta) / l_new;
            o_row[i] = o_new;
        }
        if (lane == 0) {
            m_row[warp_id] = m_new;
            l_row[warp_id] = l_new;
        }
        __syncthreads(); // ensure all warps finish before loading next tile
    }

    for (int i = lane; i < d; i += blockDim.x) {
        O[(batch * nh * N * d) + (head * N * d) + (row * d) + i] = o_row[i];
    }
}
} // namespace

torch::Tensor flash_attention_unified(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B = Q.size(0);	// Batch size
    const int nh = Q.size(1);	// Number of heads
    const int N = Q.size(2);	// Sequence size
    const int d = Q.size(3);	// Embedding size

    // Allocate using unified memory
    float *Q_um, *K_um, *V_um, *O_um;
    size_t qkv_size = B * nh * N * d * sizeof(float);
    cudaMallocManaged(&Q_um, qkv_size);
    cudaMallocManaged(&K_um, qkv_size);
    cudaMallocManaged(&V_um, qkv_size);
    cudaMallocManaged(&O_um, qkv_size);

    cudaMemcpy(Q_um, Q.data_ptr<float>(), qkv_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(K_um, K.data_ptr<float>(), qkv_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(V_um, V.data_ptr<float>(), qkv_size, cudaMemcpyDeviceToDevice);

    const float scale = 1.0f / sqrtf((float)d);
    dim3 grid(B, nh, (N + kBr - 1) / kBr);
    dim3 block(kBc, kBr);
    // Shared: q_rows (Br*d) + o_rows (Br*d) + k_tile (Bc*d) + v_tile (Bc*d)
    size_t smem = (size_t)((2 * kBr * d) + (2 * kBc * d)) * sizeof(float);

    flash_attn_online_um<kBc, kBr><<<grid, block, smem>>>(Q_um, K_um, V_um, O_um, N, d, nh, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("flash_attn_online_um launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    auto O = torch::zeros_like(Q);
    cudaMemcpy(O.data_ptr<float>(), O_um, qkv_size, cudaMemcpyDeviceToDevice);

    cudaFree(Q_um);
    cudaFree(K_um);
    cudaFree(V_um);
    cudaFree(O_um);
    return O;
}
