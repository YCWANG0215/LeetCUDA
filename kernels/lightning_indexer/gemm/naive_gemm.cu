#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// __global__ void naiveGEMM(float *A, float *B, float *C, const int M, const int K, const int N)
// {
//     // 每个thread负责C中1个元素的计算
//     int r = blockIdx.y * blockDim.y + threadIdx.y;
//     int c = blockIdx.x * blockDim.x + threadIdx.x;

//     if (r >= M || c >= N) {
//         return;
//     }

//     float val = 0.0f;
//     for (int k = 0; k < K; ++k) {
//         val += A[r * K + k] * B[k * N + c];
//     }

//     C[r * N + c] = val;
// }

// __global__ void naiveGEMM(float *A, float *B, float *C, const int M, const int K, const int N)
// {
//     // 每个thread负责输出矩阵C中的一个元素
//     int r = threadIdx.y + blockIdx.y * blockDim.y;
//     int c = threadIdx.x + blockIdx.x * blockDim.x;
//     if (r >= M || c >= N) {
//         return;
//     }

//     float val = 0.0f;
//     for (int k = 0; k < K; ++k) {
//         val += A[r * K + k] * B[k * N + c];
//     }
//     C[r * N + c] = val;
// }


__global__ void naiveGEMM(float *A, float *B, float *C, int M, int K, int N)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M || c >= N) {
        return;
    }

    float val = 0.0f;
    for (int k = 0; k < K; ++k) {
        val += A[r * K + k] * B[k * N + c];
    }
    C[r * N + c] = val;
}

int main()
{
    const int M = 5120;
    const int N = 4096;
    const int K = 4096;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    //-----------------------------------
    // Host Memory
    //-----------------------------------
    std::vector<float> hA(M * K);
    std::vector<float> hB(K * N);
    std::vector<float> hC(M * N);

    for (int i = 0; i < M * K; ++i) {
        hA[i] = 1.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        hB[i] = 1.0f;
    }

    //-----------------------------------
    // Device Memory
    //-----------------------------------
    float *dA;
    float *dB;
    float *dC;

    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice);

    //-----------------------------------
    // Launch Config
    //-----------------------------------
    // dim3 block(16, 16);
    dim3 block(32, 8);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);
    
    //-----------------------------------
    // Warmup
    //-----------------------------------
    for (int i = 0; i < 10; ++i) {
        naiveGEMM<<<grid, block>>>(dA, dB, dC, M, K, N);
    }
    cudaDeviceSynchronize();

    //-----------------------------------
    // Benchmark
    //-----------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int repeat = 10;
    for (int i = 0; i < repeat; ++i) {
        naiveGEMM<<<grid, block>>>(dA, dB, dC, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / repeat;

    //-----------------------------------
    // Copy Back
    //-----------------------------------
    cudaMemcpy(
        hC.data(),
        dC,
        sizeC,
        cudaMemcpyDeviceToHost);
    //-----------------------------------
    // FLOPS
    //-----------------------------------
    double flops =
        2.0 * M * N * K;

    double tflops =
        flops /
        (avg_ms * 1e-3) /
        1e12;

    //-----------------------------------
    // Print
    //-----------------------------------
    std::cout
        << "Average Time: "
        << avg_ms
        << " ms\n";

    std::cout
        << "TFLOPS: "
        << tflops
        << std::endl;

    //-----------------------------------
    // Free
    //-----------------------------------
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}