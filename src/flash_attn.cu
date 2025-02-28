#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void _flash_attention(const float* Q, const float* K, const float* V, float* l, float *m, float* O, 
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
        for (int i = 0; i < d; i++) { // Load K, V
            tile_K[(idx * d) + i] = K[(batch_idx * nh * N * d) + (head_idx * N * d) + (tile_size * k) + (idx * d) + i];
            tile_V[(idx * d) + i] = V[(batch_idx * nh * N * d) + (head_idx * N * d) + (tile_size * k) + (idx * d) + i];
        }
        __syncthreads(); // Synchronize after loading to SMEM

        for (int i = 0; i < row_tile; i++)  {
            for (int j = 0; j < d; j++) { // Load Q
                tile_Q[(idx * d) + j] = Q[(batch_idx * nh * N * d) + (head_idx * N * d) + (tile_size * i) + (idx * d) + j];
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

            float global_m = max(m[(batch_idx * nh * N) + (head_idx * N) + (row_block * i) + idx], local_m); // 강의교안 14p의 수식
            float global_l = l[(batch_idx * nh * N) + (head_idx * N) + (row_block * i) + idx] \
                * expf(m[(batch_idx * nh * N) + (head_idx * N) + (row_block * i) + idx] - global_m) \
                + local_l * expf(local_m - global_m);

            for (int x = 0; x < d; x++) {
                float tiled_output = 0;
                for (int y = 0; y < col_block; y++) {
                    tiled_output += tile_S[(col_block * idx) + y] * tile_V[(y * d) + x];
                }

                O[(batch_idx * nh * N * d) + (head_idx * N * d) + (tile_size * i) + (idx * d) + x] = // 강의교안 17p의 수식
                    ((O[(batch_idx * nh * N * d) + (head_idx * N * d) + (tile_size * i) + (idx * d) + x] 
                    * l[(batch_idx * nh * N) + (head_idx * N) + (row_block * i) + idx] \
                    * expf(m[(batch_idx * nh * N) + (head_idx * N) + (row_block * i) + idx] - global_m)) \
                    + (tiled_output * expf(local_m - global_m))) / global_l;
            }

            m[(batch_idx * nh * N) + (head_idx * N) + (row_block * i) + idx] = global_m;
            l[(batch_idx * nh * N) + (head_idx * N) + (row_block * i) + idx] = global_l;
        }
        __syncthreads();
    }

}

torch::Tensor flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B = Q.size(0);	// Batch size
    const int nh = Q.size(1);	// Number of heads
    const int N = Q.size(2);	// Sequence size
    const int d = Q.size(3);	// Embedding size

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);
    
    const int col_block = 32; const int row_block = 32; 
    const int col_tile = ceil((float) N / col_block); const int row_tile = ceil((float) N / row_block);
    const float scale = 1.0 / sqrt(d);

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d\n", max_sram_size);

    const int sram_size = (col_block * row_block * sizeof(float)) // For S's tile
        + (3 * col_block * d * sizeof(float)); // For Q, K, V's tile

    dim3 grid(B, nh);
    dim3 block(col_block);

    _flash_attention<<<grid, block, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>(),
        N, d, scale, col_tile, row_tile, col_block, row_block
    );

    return O;
}
