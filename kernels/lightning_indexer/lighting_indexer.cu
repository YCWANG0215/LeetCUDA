#include <cuda_runtime.h>
#define WARP_SIZE 32

template <typename QKT, typename SCORE_T> 
__global__ void lightningIndexer(QKT *q, QKT *k, float *w, int topk,
                                 int *cuSeqlensQ, int *cuSeqlensK,
                                 int *sequsedQ, int *sequsedK, int *cmpResidualK,
                                 int *blockTable, int *outputIdxOffset,
                                 int maxSeqlenQ, std::string layoutQ, std::string layoutK,
                                 int maskMode, int cmpRatio, int returnValue,
                                 int *sparseIndices, float *sparseValues)
{
    // 1. GEMM 计算S=Q*K

    // 2. W按最后一个维度广播到q的长度
    
    // 3. ReduceSum(relu(W*S))

    // 4. Radix排序，选出topk个索引
}