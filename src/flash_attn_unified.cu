#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void _flash_attention_unified(const float* Q, const float* K, const float* V, float* l, float *m, float* O,
        const int N, const int d, const float scale, const int col_tile, const int row_tile, const int col_block, const int row_block) {

    int tile_size = col_block * d;
    extern __shared__ float smem[];
    int nh = gridDim.y;
    int idx = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    float* tile_Q = smem; float* tile_K = &smem[tile_size];
    float* tile_V = &smem[tile_size * 2]; float* tile_S = &smem[tile_size * 3];

    for (int k = 0; k < col_tile; k++) {
        int global_col = (col_block * k) + idx;
        for (int i = 0; i < d; i++) { // Load K, V
            if (global_col < N) {
                tile_K[(idx * d) + i] = K[(batch_idx * nh * N * d) + (head_idx * N * d) + (global_col * d) + i];
                tile_V[(idx * d) + i] = V[(batch_idx * nh * N * d) + (head_idx * N * d) + (global_col * d) + i];
            } else {
                tile_K[(idx * d) + i] = 0.0f;
                tile_V[(idx * d) + i] = 0.0f;
            }
        }
        __syncthreads(); // Synchronize after loading to SMEM

        for (int i = 0; i < row_tile; i++)  {
            int global_row = (row_block * i) + idx;
            for (int j = 0; j < d; j++) { // Load Q
                if (global_row < N) {
                    tile_Q[(idx * d) + j] = Q[(batch_idx * nh * N * d) + (head_idx * N * d) + (global_row * d) + j];
                } else {
                    tile_Q[(idx * d) + j] = 0.0f;
                }
            }

            float local_m = -INFINITY;
            for (int y = 0; y < col_block; y++) {
                float QKT = 0.0f;
                for (int x = 0; x < d; x++) {
                    QKT += tile_Q[(idx * d) + x] * tile_K[(y * d) + x]; // Scaled Dot-Product of tiled Q and K
                }
                float Score = QKT * scale;
                tile_S[(col_block * idx) + y] = Score; // Multiply Scale to get Score

                if (Score > local_m) local_m = Score; // Reduce-Max
            }

            float local_l = 0;
            for (int y = 0; y < col_block; y++) { // Reduce-Sum-Exp
                tile_S[(col_block * idx) + y] = expf(tile_S[(col_block * idx) + y] - local_m);
                local_l += tile_S[(col_block * idx) + y];
            }

            if (global_row < N) {
                float global_m = max(m[(batch_idx * nh * N) + (head_idx * N) + global_row], local_m);
                float global_l = l[(batch_idx * nh * N) + (head_idx * N) + global_row] \
                    * expf(m[(batch_idx * nh * N) + (head_idx * N) + global_row] - global_m) \
                    + local_l * expf(local_m - global_m);

                for (int x = 0; x < d; x++) {
                    float tiled_output = 0;
                    for (int y = 0; y < col_block; y++) {
                        tiled_output += tile_S[(col_block * idx) + y] * tile_V[(y * d) + x];
                    }

                    O[(batch_idx * nh * N * d) + (head_idx * N * d) + (global_row * d) + x] =
                        ((O[(batch_idx * nh * N * d) + (head_idx * N * d) + (global_row * d) + x]
                        * l[(batch_idx * nh * N) + (head_idx * N) + global_row] \
                        * expf(m[(batch_idx * nh * N) + (head_idx * N) + global_row] - global_m)) \
                        + (tiled_output * expf(local_m - global_m))) / global_l;
                }

                m[(batch_idx * nh * N) + (head_idx * N) + global_row] = global_m;
                l[(batch_idx * nh * N) + (head_idx * N) + global_row] = global_l;
            }
        }
        __syncthreads();
    }

}

torch::Tensor flash_attention_unified(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B = Q.size(0);	// Batch size
    const int nh = Q.size(1);	// Number of heads
    const int N = Q.size(2);	// Sequence size
    const int d = Q.size(3);	// Embedding size

    // Allocate using unified memory instead of regular CUDA memory
    float *Q_um, *K_um, *V_um, *O_um, *l_um, *m_um;
    size_t qkv_size = B * nh * N * d * sizeof(float);
    size_t l_size = B * nh * N * sizeof(float);

    cudaMallocManaged(&Q_um, qkv_size);
    cudaMallocManaged(&K_um, qkv_size);
    cudaMallocManaged(&V_um, qkv_size);
    cudaMallocManaged(&O_um, qkv_size);
    cudaMallocManaged(&l_um, l_size);
    cudaMallocManaged(&m_um, l_size);

    // Copy input data to unified memory
    cudaMemcpy(Q_um, Q.data_ptr<float>(), qkv_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(K_um, K.data_ptr<float>(), qkv_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(V_um, V.data_ptr<float>(), qkv_size, cudaMemcpyDeviceToDevice);

    // Initialize O, l, m
    cudaMemset(O_um, 0, qkv_size);
    cudaMemset(l_um, 0, l_size);
    for (int i = 0; i < B * nh * N; i++) {
        m_um[i] = -INFINITY;
    }

    const int col_block = 32; const int row_block = 32;
    const int col_tile = ceil((float) N / col_block); const int row_tile = ceil((float) N / row_block);
    const float scale = 1.0 / sqrt(d);

    const int sram_size = (col_block * row_block * sizeof(float)) // For S's tile
        + (3 * col_block * d * sizeof(float)); // For Q, K, V's tile

    dim3 grid(B, nh);
    dim3 block(col_block);

    _flash_attention_unified<<<grid, block, sram_size>>>(
        Q_um, K_um, V_um,
        l_um, m_um, O_um,
        N, d, scale, col_tile, row_tile, col_block, row_block
    );

    cudaDeviceSynchronize();

    // Copy result back to torch tensor
    auto O = torch::zeros_like(Q);
    cudaMemcpy(O.data_ptr<float>(), O_um, qkv_size, cudaMemcpyDeviceToDevice);

    // Free unified memory
    cudaFree(Q_um);
    cudaFree(K_um);
    cudaFree(V_um);
    cudaFree(O_um);
    cudaFree(l_um);
    cudaFree(m_um);

    return O;
}
