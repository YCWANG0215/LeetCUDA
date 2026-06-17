/**
    对32*32的矩阵进行转置操作
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

/**
    NaiveRow:
        ix -> c, iy -> r
        A[r][c] = A[r * N + c] = A[iy * N + ix]
        B[c][r] = B[c * M + r] = B[ix * M + iy]
 */
__global__ void transposeNaiveRow(float* A, float* B, const int M, const int N)
{
    int ix = threadIdx.x + blockNum.x * blockDim.x;
    int iy = threadIdx.y + blockNum.y * blockDim.y;
    
    if (ix < N && iy < M) {
        B[ix * N + iy] = A[iy * M + ix];
    }
}

/**
    NaiveCol:
        ix -> r, iy -> c
        A[r][c] = A[r * N + c] = A[ix * N + iy]
        B[c][r] = B[c * M + r] = B[iy * M + ix]
 */
__global__ void transposeNaiveCol(float* A, float* B, const int M, const int N)
{
    int ix = threadIdx.x + blockNum.x * blockDim.x;
    int iy = threadIdx.y + blockNum.y * blockDim.y;

    if (ix < M && iy < N) {
        B[iy * M + ix] = A[ix * N + iy];
    }
}

template<int Bm, int Bn>
__global__ void transposeColNElements(float* A, float* B, const int M, const int N)
{
    // (r0, c0)表示tile内左上角元素的坐标
    int r0 = blockIdx.x * Bm;
    int c0 = blockIdx.y * Bn;
    
    #pragma unroll
    for (int x = threadIdx.x; x < Bm; x += blockDim.x) {
        int r = r0 + x;
        if (r >= M) {
            return;
        }
        #pragma unroll
        for (int y = threadIdx.y; y < Bn; y += blockDim.y) {
            int c = c0 + y;
            if (c < N) {
                B[c * M + r] = A[r * N + c];
            }
        }
    }
}

// template<int Bm, int Bn>
// __global__ void transposeShared(float *A, float *B, const int M, const int N)
// {
//     __shared__ float tile[Bm][Bn];
//     /* -------- 读取阶段 -------- */
//     // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
//     int r0 = blockIdx.y * Bm;
//     int c0 = blockIdx.x * Bn;

//     // thread y 方向负责：矩阵 A 的行，shared memory 的行
//     // thread x 方向负责：矩阵 A 的列，shared memory 的列
//     // shared memory 中的元素 tile[y][x] = A[r0 + y, c0 + x]
//     #pragma unroll
//     for (int y = threadIdx.y; y < Bm; y += blockDim.y) {
//         int r = r0 + y;
//         if (r >= M) {
//             break;
//         }
//         #pragma unroll
//         for (int x = threadIdx.x; x < Bn; x += blockDim.x) {
//             int c = c0 + x;
//             if (c < N) {
//                 tile[y][x] = A[r * N + c];
//             }
//         }
//     }
//     __syncthreads();

//     /* -------- 写入阶段 -------- */
//     // (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
//     // thread y 方向负责：矩阵 B 的行，shared memory 的列
//     // thread x 方向负责：矩阵 B 的列，shared memory 的行
//     // shared memory 中的元素 tile[x][y] = B[c0 + y, r0 + x]
//     #pragma unroll
//     for (int y = threadIdx.y; y < Bn; y += blockDim.y) {
//         int c = c0 + y;
//         if (c >= N) {
//             break;
//         }
//         #pragma unroll
//         for (int x = threadIdx.x; x < Bm; x += blockDim.x) {
//             int r = r0 + x;
//             if (r < M) {
//                 B[c * M + r] = tile[x][y];
//             }
//         }
//     }
// }

template<int Bm, int Bn>
__global__ void transposeShared(float *A, float *B, const int M, const int N)
{
    __shared__ float tile[Bm][Bn];

    int r0 = Bm * blockDim.y;
    int c0 = Bn * blockDim.x;

    // 1. 将A中内容写入shared memory
    for (int y = threadIdx.y; y < Bm; y += blockDim.y) {
        int r = r0 + y;
        if (r >= M) {
            return;
        }
        for (int x = threadIdx.x; x < Bn; x += blockDim.x) {
            int c = c0 + x;
            if (c < N) {
                tile[y][x] = A[r0 * N + c];
            }
        }
    }

    __syncthreads();

    // 2. 将shared memory中的内容写入B
    for (int y = threadIdx.y; y < Bn; y += blockDim.y) {
        int c = c0 + y;
        if (c >= N) {
            break;
        }
        for (int x = threadIdx.x; x < Bm; x += blockDim.x) {
            int r = r0 + x;
            if (r < M) {
                B[c * M + r] = tile[x][y];
            }
        }
    }
}