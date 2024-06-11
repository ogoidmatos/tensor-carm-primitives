#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <vector>

#include "../../../tc-benchmark/nvml_tools.cu"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

// #define POWER

// The code section below describes datatype for input, output matrices and
// computation between elements in input matrices.
using ElementAccumulator = float;  // <- data type of accumulator
using ElementComputeEpilogue =
    ElementAccumulator;       // <- data type of epilogue operations
using ElementInputA = float;  // <- data type of elements in input matrix A
using ElementInputB = float;  // <- data type of elements in input matrix B
using ElementOutput = float;  // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices.
// Row Major for Matrix A, Column Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;  // <-
                                                                        // ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,  // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::
              value,          // <- the number of elements per vectorized
                              // memory access. For a byte, it's 16
                              // elements. This becomes the vector width of
                              // math instructions in the epilogue too
    ElementAccumulator,       // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear
                              // combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

// Split K dimension into 1 partitions
constexpr int split_k_slices = 1;

using Gemm = cutlass::gemm::device::GemmArray<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

using data_type = float;

int run(int m, int n, int k, int batch) {
  std::thread measuring_thread;
  monitor_args thread_args;
  thread_args.powerArray = std::vector<int>();
  thread_args.clockArray = std::vector<int>();
  thread_args.flag = 0;

  init_nvml(&thread_args, &measuring_thread);
  cudaDeviceSynchronize();

  const int length_m = m;
  const int length_n = n;
  const int length_k = k;

  std::vector<std::vector<data_type>> A(
      batch, std::vector<data_type>(length_m * length_k, 1.0 / batch));
  std::vector<std::vector<data_type>> B(
      batch, std::vector<data_type>(length_k * length_n, 1.0 / batch));
  std::vector<std::vector<data_type>> C(
      batch, std::vector<data_type>(length_m * length_n, 0.0));

  const data_type alpha = 1.0f;
  const data_type beta = 0.0f;

  data_type **d_A_array = nullptr;
  data_type **d_B_array = nullptr;
  data_type **d_C_array = nullptr;

  std::vector<data_type *> d_A(batch, nullptr);
  std::vector<data_type *> d_B(batch, nullptr);
  std::vector<data_type *> d_C(batch, nullptr);

  for (int i = 0; i < batch; i++) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A[i]),
                          sizeof(data_type) * A[i].size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B[i]),
                          sizeof(data_type) * B[i].size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C[i]),
                          sizeof(data_type) * C[i].size()));
  }

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_array),
                        sizeof(data_type *) * batch));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B_array),
                        sizeof(data_type *) * batch));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_array),
                        sizeof(data_type *) * batch));

  for (int i = 0; i < batch; i++) {
    CUDA_CHECK(cudaMemcpy(d_A[i], A[i].data(), sizeof(data_type) * A[i].size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B[i], B[i].data(), sizeof(data_type) * B[i].size(),
                          cudaMemcpyHostToDevice));
  }

  CUDA_CHECK(cudaMemcpy(d_A_array, d_A.data(), sizeof(data_type *) * batch,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B_array, d_B.data(), sizeof(data_type *) * batch,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C_array, d_C.data(), sizeof(data_type *) * batch,
                        cudaMemcpyHostToDevice));

  // // Create a tuple of gemm kernel arguments. This is later passed as
  // arguments
  // // to launch instantiated CUTLASS kernel
  // typename Gemm::Arguments arguments{
  //     {length_m, length_n,
  //      length_k},  // <- problem size of matrix multiplication
  //     d_A_array,
  //     length_m,  // <- reference to matrix A on device
  //     d_B_array,
  //     length_n,  // <- reference to matrix B on device
  //     d_C_array,
  //     length_k,  // <- reference to matrix C on device
  //     d_C_array,
  //     length_k,       // <- reference to matrix D on device
  //     {alpha, beta},  // <- tuple of alpha and beta
  //     split_k_slices,
  //     batch};  // <- k-dimension split factor

  // // Using the arguments, query for extra workspace required for matrix
  // // multiplication computation
  // size_t workspace_size = Gemm::get_workspace_size(arguments);

  // // Allocate workspace memory
  // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  // printf("Workspace size: %lu\n", workspace_size);
  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // // Check the problem size is supported or not
  // cutlass::Status status = gemm_op.can_implement(arguments);
  // CUTLASS_CHECK(status);
  // printf("Problem size is supported\n");
  // // Initialize CUTLASS kernel with arguments and workspace pointer
  // status = gemm_op.initialize(arguments, workspace.get());
  // CUTLASS_CHECK(status);
  printf("Initialized\n");
  thread_args.flag = 1;
#ifdef POWER
#pragma unroll
  for (int i = 0; i < 32768 * 64; i++)
#endif
    // Launch initialized CUTLASS kernel
    cutlass::Status status =
        gemm_op({{length_m, length_n,
                  length_k},  // <- problem size of matrix multiplication
                 d_A_array,
                 length_m,  // <- reference to matrix A on device
                 d_B_array,
                 length_n,  // <- reference to matrix B on device
                 d_C_array,
                 length_k,  // <- reference to matrix C on device
                 d_C_array,
                 length_k,       // <- reference to matrix D on device
                 {alpha, beta},  // <- tuple of alpha and beta
                 batch});
  cudaDeviceSynchronize();
  thread_args.flag = 0;
  stop_nvml(&measuring_thread, thread_args.powerArray, thread_args.clockArray);

  CUTLASS_CHECK(status);
  printf("Launched\n");

  for (int i = 0; i < batch; i++) {
    CUDA_CHECK(cudaMemcpy(C[i].data(), d_C[i], sizeof(data_type) * C[i].size(),
                          cudaMemcpyDeviceToHost));
  }

  /* free resources */
  CUDA_CHECK(cudaFree(d_A_array));
  CUDA_CHECK(cudaFree(d_B_array));
  CUDA_CHECK(cudaFree(d_C_array));
  for (int i = 0; i < batch; i++) {
    CUDA_CHECK(cudaFree(d_A[i]));
    CUDA_CHECK(cudaFree(d_B[i]));
    CUDA_CHECK(cudaFree(d_C[i]));
  }

  printf("C[0][0] = %f\n", C[0][0]);
  return 0;
}

int main(int argc, char *argv[]) {
  int m = 1024;
  int n = 1024;
  int k = 1024;

  int batch_size = 1;

  int c;
  cudaSetDevice(0);
  while ((c = getopt(argc, argv, "m:n:k:a:b:h")) != -1) switch (c) {
      case 'a':
        m = n = k = atoi(optarg);
        break;
      case 'm':
        m = atoi(optarg);
        break;
      case 'n':
        n = atoi(optarg);
        break;
      case 'k':
        k = atoi(optarg);
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      case 'h':
        fprintf(stdout,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=1024]\n\t-n \t N "
                "dimension [int] [default=1024]\n\t-k \t K dimension [int] "
                "[default=1024]\n\t-a \t All "
                "dimensions [int]\n\t-b \t Batch Size [int] [default=1]\n\n",
                argv[0]);
        exit(EXIT_SUCCESS);
      default:
        fprintf(stderr,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=1024]\n\t-n \t N "
                "dimension [int] [default=1024]\n\t-k \t K dimension [int] "
                "[default=1024]\n\t-a \t All "
                "dimensions [int]\n\t-b \t Batch Size [int] [default=1]\n\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

  printf("GEMM with dimensions m=%d, n=%d, k=%d\nBatch Size: %d\n", m, n, k,
         batch_size);

  bool notSupported = false;

  // Turing Tensor Core operations exposed with mma.sync and ldmatrix are
  // first available in CUDA 10.2.
  //
  // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ > 10 ||
        (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
    std::cerr << "Turing Tensor Core operations must be compiled with CUDA "
                 "10.2 Toolkit or later."
              << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: "
              << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with "
                 "compute capability at least 80."
              << std::endl;

    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are
    // no-op.
    return 0;
  }

  return run(m, n, k, batch_size);
}