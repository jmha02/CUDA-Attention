#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void scaled_dot_product_unified(const float* Q, const float* K, float* S, int B, int nh, int N, int d) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tiles_per_row = (N + blockDim.x - 1) / blockDim.x;
    int tile_row = blockIdx.z / tiles_per_row;
    int tile_col = blockIdx.z % tiles_per_row;
    int row = threadIdx.y + tile_row * blockDim.y;
    int col = threadIdx.x + tile_col * blockDim.x;

    if (batch_idx < B && head_idx < nh && row < N && col < N) {
        float scale = 1.0f / sqrtf((float)d);
        float value = 0.0f;

        for (int i = 0; i < d; i++) {
            value += Q[batch_idx * nh * N * d + head_idx * N * d + row * d + i] *
                     K[batch_idx * nh * N * d + head_idx * N * d + col * d + i];
        }
        int idx = batch_idx * nh * N * N + head_idx * N * N + row * N + col;
        S[idx] = value * scale;
    }
}

__global__ void reduce_max_unified(float* S, float* M, int B, int nh, int N) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = threadIdx.x;

    if (batch_idx < B && head_idx < nh && row < N) {
        int idx = batch_idx * nh * N * N + head_idx * N * N + row * N;

        float m = -INFINITY;
        for (int i = 0; i < N; ++i) {
            m = max(m, S[idx + i]);
        }
        M[batch_idx * nh * N + head_idx * N + row] = m;
    }
}

__global__ void reduce_sum_exp_unified(float* S, float* M, float* L, int B, int nh, int N) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = threadIdx.x;

    if (batch_idx < B && head_idx < nh && row < N) {
        int idx = batch_idx * nh * N * N + head_idx * N * N + row * N;

        float l = 0.0f;
        for (int i = 0; i < N; ++i) {
            S[idx + i] = expf(S[idx + i] - M[batch_idx * nh * N + head_idx * N + row]);
            l += S[idx + i];
        }
        L[batch_idx * nh * N + head_idx * N + row] = l;
    }
}

__global__ void softmax_unified(float* S, float* L, int B, int nh, int N) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = threadIdx.x;

    if (batch_idx < B && head_idx < nh && row < N) {
        int idx = batch_idx * nh * N * N + head_idx * N * N + row * N;

        for (int i = 0; i < N; ++i) {
            S[idx + i] /= L[batch_idx * nh * N + head_idx * N + row];
        }
    }
}

__global__ void weighted_sum_unified(const float* S, const float* V, float* O, int B, int nh, int N, int d) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tiles_per_row = (d + blockDim.x - 1) / blockDim.x;
    int tile_row = blockIdx.z / tiles_per_row;
    int tile_col = blockIdx.z % tiles_per_row;
    int row = threadIdx.y + tile_row * blockDim.y;
    int col = threadIdx.x + tile_col * blockDim.x;

    if (batch_idx < B && head_idx < nh && row < N && col < d) {
        float value = 0.0f;

        for (int i = 0; i < N; i++) {
            value += S[batch_idx * nh * N * N + head_idx * N * N + row * N + i] *
                     V[batch_idx * nh * N * d + head_idx * N * d + i * d + col];
        }
        O[batch_idx * nh * N * d + head_idx * N * d + row * d + col] = value;
    }
}

torch::Tensor naive_attention_unified(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B = Q.size(0);  // Batch size
    const int nh = Q.size(1);  // Number of heads
    const int N = Q.size(2);  // Sequence size
    const int d = Q.size(3);  // Key/Query dimension

    // Allocate using unified memory
    float *Q_um, *K_um, *V_um, *O_um, *M_um, *L_um, *S_um;
    size_t qkv_size = B * nh * N * d * sizeof(float);
    size_t ml_size = B * nh * N * sizeof(float);
    size_t s_size = B * nh * N * N * sizeof(float);

    cudaMallocManaged(&Q_um, qkv_size);
    cudaMallocManaged(&K_um, qkv_size);
    cudaMallocManaged(&V_um, qkv_size);
    cudaMallocManaged(&O_um, qkv_size);
    cudaMallocManaged(&M_um, ml_size);
    cudaMallocManaged(&L_um, ml_size);
    cudaMallocManaged(&S_um, s_size);

    // Copy input data to unified memory
    cudaMemcpy(Q_um, Q.data_ptr<float>(), qkv_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(K_um, K.data_ptr<float>(), qkv_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(V_um, V.data_ptr<float>(), qkv_size, cudaMemcpyDeviceToDevice);

    // Initialize outputs
    cudaMemset(O_um, 0, qkv_size);
    cudaMemset(M_um, 0, ml_size);
    cudaMemset(L_um, 0, ml_size);
    cudaMemset(S_um, 0, s_size);

    dim3 block(32, 32); // Max Thread
    dim3 smgrid(B, nh); // For Softmax
    int tiles_N = (N + block.x - 1) / block.x;
    int tiles_N_rows = (N + block.y - 1) / block.y;
    int tiles_d = (d + block.x - 1) / block.x;
    dim3 mmgrid_scores(B, nh, tiles_N * tiles_N_rows); // For S
    dim3 mmgrid_out(B, nh, tiles_N_rows * tiles_d); // For O

    scaled_dot_product_unified<<<mmgrid_scores, block>>>(Q_um, K_um, S_um, B, nh, N, d);
    reduce_max_unified<<<smgrid, N>>>(S_um, M_um, B, nh, N);
    reduce_sum_exp_unified<<<smgrid, N>>>(S_um, M_um, L_um, B, nh, N);
    softmax_unified<<<smgrid, N>>>(S_um, L_um, B, nh, N);
    weighted_sum_unified<<<mmgrid_out, block>>>(S_um, V_um, O_um, B, nh, N, d);

    cudaDeviceSynchronize();

    // Copy result back to torch tensor
    auto O = torch::zeros_like(Q);
    cudaMemcpy(O.data_ptr<float>(), O_um, qkv_size, cudaMemcpyDeviceToDevice);

    // Free unified memory
    cudaFree(Q_um);
    cudaFree(K_um);
    cudaFree(V_um);
    cudaFree(O_um);
    cudaFree(M_um);
    cudaFree(L_um);
    cudaFree(S_um);

    return O;
}
