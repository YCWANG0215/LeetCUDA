#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32

/**
 * reinterpret_cast<T>：把指针强制解释成类型T。以reinterpret_cast<float4 *>为例，
 * 也就是从当前位置开始，把接下来的16个字节（4个float）当做一个float4来看
 */
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
/**
 * FP32：单精度浮点数
 * 逐元素相加，每次处理一个float
 */
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c,
                                           int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}


__global__ void elementwise_add_fp32_kernel_self(float *a, float *b, float *c, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
/**
 * 使用CUDA内建类型float4，一次处理4个float32（128b）
 * float4本质上是一个包含4个float的结构体，带有16字节对齐要求，这样GPU可以用128bit/load指令一次性加载/存储4个浮点数
 */
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  }
}

__global__ void elementwise_add_f32x4_kernel_self(float *a, float *b, float *c, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_a = reinterpret_cast<float4 *>(&a[idx])[0];
    float4 reg_b = reinterpret_cast<float4 *>(&b[idx])[0];
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_b.w = reg_a.w + reg_b.w;
    reinterpret_cast<float4 *>(&c[idx])[0] = reg_c;
  }

}

// FP16
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
/**
 * FP16（即CUDA的__half）不是C++标准类型，在C++中直接写两个__half类型的加法是不合法的，因为+运算符没有为__half定义
 * 所以CUDA提供了如__hadd这样的intrisics（内建函数）来对half类型进行操作
 */
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);
}


__global__ void elementwise_add_f16_kernel_self(half *a, half *b, half *c, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    c[idx] = __hadd(a[idx], b[idx]);
  }
}

// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;
    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    HALF2(c[idx]) = reg_c;
  }
}


__global__ void elementwise_add_f16x2_kernel_self(half *a, half *b, half *c, int N) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
  if (idx < N) {
    half2 reg_a = reinterpret_cast<half2 *>(&a[idx])[0];
    half2 reg_b = reinterpret_cast<half2 *>(&b[idx])[0];
    half2 reg_c;
    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    reinterpret_cast<half2 *>(&c[idx])[0] = reg_c;
  }
}


// __global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
//   int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
//   half2 reg_a_0 = HALF2(a[idx + 0]);
//   half2 reg_a_1 = HALF2(a[idx + 2]);
//   half2 reg_a_2 = HALF2(a[idx + 4]);
//   half2 reg_a_3 = HALF2(a[idx + 6]);
//   half2 reg_b_0 = HALF2(b[idx + 0]);
//   half2 reg_b_1 = HALF2(b[idx + 2]);
//   half2 reg_b_2 = HALF2(b[idx + 4]);
//   half2 reg_b_3 = HALF2(b[idx + 6]);
//   half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
//   reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
//   reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
//   reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
//   reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
//   reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
//   reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
//   reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
//   reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
//   if ((idx + 0) < N) {
//     HALF2(c[idx + 0]) = reg_c_0;
//   }
//   if ((idx + 2) < N) {
//     HALF2(c[idx + 2]) = reg_c_1;
//   }
//   if ((idx + 4) < N) {
//     HALF2(c[idx + 4]) = reg_c_2;
//   }
//   if ((idx + 6) < N) {
//     HALF2(c[idx + 6]) = reg_c_3;
//   }
// }

/**
 * Self Version
 */
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (idx < N) {
    half2 reg_a_0 = reinterpret_cast<half2 *>(&a[idx + 0])[0];
    half2 reg_a_1 = reinterpret_cast<half2 *>(&a[idx + 2])[0];
    half2 reg_a_2 = reinterpret_cast<half2 *>(&a[idx + 4])[0];
    half2 reg_a_3 = reinterpret_cast<half2 *>(&a[idx + 6])[0];
    half2 reg_b_0 = reinterpret_cast<half2 *>(&b[idx + 0])[0];
    half2 reg_b_1 = reinterpret_cast<half2 *>(&b[idx + 2])[0];
    half2 reg_b_2 = reinterpret_cast<half2 *>(&b[idx + 4])[0];
    half2 reg_b_3 = reinterpret_cast<half2 *>(&b[idx + 6])[0];

    // half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
    // reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
    // reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
    // reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
    // reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
    // reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
    // reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
    // reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
    // reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
    half2 reg_c_0 = __hadd2(reg_a_0, reg_b_0);
    half2 reg_c_1 = __hadd2(reg_a_1, reg_b_1);
    half2 reg_c_2 = __hadd2(reg_a_2, reg_b_2);
    half2 reg_c_3 = __hadd2(reg_a_3, reg_b_3);
    if ((idx + 0) < N) {
      reinterpret_cast<half2 *>(&c[idx + 0])[0] = reg_c_0;
    }
    if ((idx + 2) < N) {
      reinterpret_cast<half2 *>(&c[idx + 2])[0] = reg_c_1;
    }
    if ((idx + 4) < N) {
      reinterpret_cast<half2 *>(&c[idx + 4])[0] = reg_c_2;
    }
    if ((idx + 6) < N) {
      reinterpret_cast<half2 *>(&c[idx + 6])[0] = reg_c_3;
    }
  }
}



  /**
   * #pragma unroll -> 提示编译器循环展开，所谓循环展开，就是把循环内部的操作复制多份，减少循环控制开销（判断、加法、跳转等），并为编译器创造优化机会（指令并行，寄存器复用等）
   * 如果unroll后面没有参数，编译器会尝试进行完全循环展开（如果循环次数是常量且不太大）
   * 如果unroll后面有参数，即#pragma unroll x的格式，如#pragma unroll 4会强制编译器展开4次，如果循环次数N大于4，就会部分展开（loop unrolling + leftover loop）
   * #pragma unroll 1表示不要展开。
   */
// __global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c,
//                                                   int N) {
//   int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
//   // temporary register(memory), .local space in ptx, addressable
//   half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.
//   // reinterpret as float4 and load 128 bits in 1 memory issue.
//   // float4本质上是4个float32组成的结构体，总大小是128b，且要求16字节对齐
//   // 为什么LDST128BITS宏会用float4来加载half数组？这是硬件对齐/打包优化的技巧
//   // half pack_a[8]逻辑上是8个half，而物理存储上就是连续的128bit内存块，
//   // 但CUDA硬件不直接支持一次性从内存中取8个half(128bit)作为一个SIMD向量寄存器，
//   // 而支持一次load/store 128bit，通常是float4/int4这样的类型，
//   // 所以宏LDST128BITS就是把half[8]当做float4（128bit容器）来一次性搬运，并没有把half转成float32，
//   // 只是借用float4的128bit容量来打包传输。
//   // reinterpret_cast不会做数值转换，只是告诉编译器，把这段128bit内存当做float4看。
//   // reinterpret_cast只是桥梁，算力层用half（__hadd2的输入输出），搬运层用float4
//   LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
//   LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits

// #pragma unroll
//   for (int i = 0; i < 8; i += 2) {
//     // __hadd2 for half2 x 4
//     HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
//   }
//   // reinterpret as float4 and store 128 bits in 1 memory issue.
//   if ((idx + 7) < N) {
//     LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
//   }
// }

//self version
__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
  half pack_a[8], pack_b[8], pack_c[8];
  reinterpret_cast<float4 *>(&pack_a)[0] = reinterpret_cast<float4 *>(&a[idx])[0];
  reinterpret_cast<float4 *>(&pack_b)[0] = reinterpret_cast<float4 *>(&b[idx])[0];


  for (int i = 0; i < 8; i += 2) {
    reinterpret_cast<half2 *>(&pack_c[i])[0] = __hadd2(reinterpret_cast<half2 *>(&pack_a[i])[0],
                                                       reinterpret_cast<half2 *>(&pack_b[i])[0]);
  }
  if ((idx + 7) < N) {
    reinterpret_cast<float4 *>(&c[idx])[0] = reinterpret_cast<float4 *>(&pack_c[0])[0];
  }
}


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

  /**
   * params:
   * packed_type: 拼到函数名上，用来区分不同实现
   * th_type：期望的Torch dtype，如torch::kFloat32，用来做类型检查
   * element_type：设备端元素类型指针要转换到的C++类型，如float或half
   * n_elements：每个线程一次处理的元素个数（标量=1，half2=2)，直接影响block大小的计算
   * 
   * 1. ndim != 2的情况：
   * 把所有维度元素数相乘得到总元素数N，按每块覆盖256个元素的策略配置：
   *    1.1 block = 256 / n_element --> 线程数*每线程处理的元素数 = 256
   * 2. ndim == 2的情况：即二维矩阵。令S = size(0), K = size(1), N = S*K
   *    2.1 若 K / n_element <= 1024：一行一个block的映射（每个block覆盖一行的所有列）
   */
#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                 \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}


/**
 * CUDA中Block和Grid的作用
 * BlocKDim：一个Block内有多少线程。线程数 = blockDim.x * blockDim.y * blockDim.z
 * gridDim：整个Grid里有多少Block。
 * 每个线程会有一个全局索引：int idx = blockDim.x * blockIdx.x + threadIdx.x（仅考虑一维情况）
 * 因此，合理选择Block和Grid的大小，会直接影响：
 * 1. 是否能覆盖所有元素N
 * 2. 线程调度是否高效
 * 3. 是否触发Warp对齐和Coalesced Memory Access
 * 
 * Warp是最小调度单位，其大小是32个线程，每次调度必然是32个线程一起执行。
 * 一个Block的最大线程数是1024，这些线程会被划分为多个Warp，每32个线程形成一个Warp。
 * 
 * 以宏TORCH_BINDING_ELEM_ADD中对Block和Grid的设计为例：
 * 情况A：ndim != 2，即通用多维张量展平
 *    dim3 block(256 / n_element)
 *    dim3 grid((N + 256 - 1) / 256)
 *    意为：每个线程处理n_element个元素，每个Block处理256个元素，grid的大小保证所有元素N都能被覆盖
 *    这样做可以保持Block内工作量固定（256个元素），简化调度。
 * 情况B：ndim == 2，且K / n_elements <= 1024。张量大小为[S, K]
 *    如果K / n_elements <= 1024，说明一行（K维度）能放进一个Block
 *    那么设计方式是：每个Block负责处理一整行（K个元素），Grid大小为S（每行对应一个Block）
 * 情况C：ndim == 2，且K / n_elements > 1024
 *    当一行太长，单行放不进一个Block时，就退回情况A的通用策略，这样即使行太长，仍然可以覆盖整个[S, K]矩阵
 */