#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void scaled_dot_product(const float* Q, const float* K, float* S, int B, int nh, int N, int d) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int mat_idx = blockIdx.z;
    int row = threadIdx.y + (mat_idx / 2) * (N / 2);
    int col = threadIdx.x + (mat_idx % 2) * (N / 2);

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

__global__ void reduce_max(float* S, float* M, int B, int nh, int N) {
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

__global__ void reduce_sum_exp(float* S, float* M, float* L, int B, int nh, int N) {
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

__global__ void softmax(float* S, float* L, int B, int nh, int N) {
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

__global__ void weighted_sum(const float* S, const float* V, float* O, int B, int nh, int N, int d) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int mat_idx = blockIdx.z;
    int row = threadIdx.y + (mat_idx / 2) * (N / 2);
    int col = threadIdx.x + (mat_idx % 2) * (d / 2);

    if (batch_idx < B && head_idx < nh && row < N && col < d) {
        float value = 0.0f;

        for (int i = 0; i < N; i++) {
            value += S[batch_idx * nh * N * N + head_idx * N * N + row * N + i] *
                     V[batch_idx * nh * N * d + head_idx * N * d + i * d + col];
        }
        O[batch_idx * nh * N * d + head_idx * N * d + row * d + col] = value;
    }
}

torch::Tensor naive_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B = Q.size(0);  // Batch size
    const int nh = Q.size(1);  // Number of heads
    const int N = Q.size(2);  // Sequence size
    const int d = Q.size(3);  // Key/Query dimension

    auto options = torch::TensorOptions().dtype(Q.dtype()).device(torch::kCUDA);
    auto M = torch::zeros({B, nh, N}, options); // For Reduce Max
    auto L = torch::zeros({B, nh, N}, options); // For Reduced Sum
    auto S = torch::zeros({B, nh, N, N}, options); // Score
    auto O = torch::zeros({B, nh, N, d}, options); // Output

    dim3 block(32, 32); // Max Thread
    dim3 smgrid(B, nh); // For Softmax
    dim3 mmgrid(B, nh, 4); // For Matmuls

    scaled_dot_product<<<mmgrid, block>>>(Q.data_ptr<float>(), K.data_ptr<float>(), S.data_ptr<float>(), B, nh, N, d);
    reduce_max<<<smgrid, N>>>(S.data_ptr<float>(), M.data_ptr<float>(), B, nh, N);
    reduce_sum_exp<<<smgrid, N>>>(S.data_ptr<float>(), M.data_ptr<float>(), L.data_ptr<float>(), B, nh, N);
    softmax<<<smgrid, N>>>(S.data_ptr<float>(), L.data_ptr<float>(), B, nh, N);
    weighted_sum<<<mmgrid, block>>>(S.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), B, nh, N, d);

    return O;
}