#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

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
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

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

using Gemm = cutlass::gemm::device::GemmSplitKParallel<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp>;

int run(int m, int n, int k, int split_k_slices) {
  const int length_m = m;
  const int length_n = n;
  const int length_k = k;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  printf("Problem size: %d x %d x %d\n", length_m, length_n, length_k);
  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  // cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
  //     problem_size.mn());  // <- Create matrix D with dimensions M x N used
  //     to
  // store output from reference kernel
  printf("Tensors initialized\n");
  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFill(tensor_a.host_view());
  cutlass::reference::host::TensorFill(tensor_b.host_view());
  cutlass::reference::host::TensorFill(tensor_c.host_view());
  // cutlass::reference::host::TensorFill(
  //     tensor_ref_d.host_view());  // <- fill matrix D for reference on host
  //     with
  // zeros
  printf("Tensors filled\n");
  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  // tensor_ref_d.sync_device();
  printf("Data copied to device\n");
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      problem_size,           // <- problem size of matrix multiplication
      tensor_a.device_ref(),  // <- reference to matrix A on device
      tensor_b.device_ref(),  // <- reference to matrix B on device
      tensor_c.device_ref(),  // <- reference to matrix C on device
      tensor_c.device_ref(),  // <- reference to matrix D on device
      {alpha, beta},          // <- tuple of alpha and beta
      split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  printf("Workspace size: %lu\n", workspace_size);
  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  printf("Problem size is supported\n");
  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  printf("Initialized\n");
  // Launch initialized CUTLASS kernel
  auto start = std::chrono::high_resolution_clock::now();
  status = gemm_op();
  CUTLASS_CHECK(status);
  // Create instantiation for device reference gemm kernel
  // cutlass::reference::device::Gemm<
  //     ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
  //     ElementOutput, LayoutOutput, ElementComputeEpilogue,
  //     ElementComputeEpilogue> gemm_device;

  // // Launch device reference gemm kernel
  // gemm_device(problem_size, alpha, tensor_a.device_ref(),
  // tensor_b.device_ref(),
  //             beta, tensor_c.device_ref(), tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Launched\n");
  printf("Time taken: %lu ns\n", duration);

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_c.sync_host();
  // tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  // bool passed = cutlass::reference::host::TensorEquals(
  //     tensor_c.host_view(), tensor_ref_d.host_view());

  // std::cout << (passed ? "Passed" : "Failed") << std::endl;

  // return (passed ? 0 : -1);
  return tensor_c.host_ref().at({0, 0});
}

int main(int argc, char *argv[]) {
  int m = 4096;
  int n = 4096;
  int k = 4096;

  int split_k_slices = 1;

  int c;
  cudaSetDevice(0);
  while ((c = getopt(argc, argv, "m:n:k:a:s:h")) != -1) switch (c) {
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
      case 's':
        split_k_slices = atoi(optarg);
        break;
      case 'h':
        fprintf(stdout,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=4096]\n\t-n \t N "
                "dimension [int] [default=4096]\n\t-k \t K dimension [int] "
                "[default=4096]\n\t-a \t All "
                "dimensions [int]\n\t-n Number of Stages [int] "
                "[default=3]\n\t-s Split K slices [int] [default=1]\n",
                argv[0]);
        exit(EXIT_SUCCESS);
      default:
        fprintf(stdout,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=4096]\n\t-n \t N "
                "dimension [int] [default=4096]\n\t-k \t K dimension [int] "
                "[default=4096]\n\t-a \t All "
                "dimensions [int]\n\t-n Number of Stages [int] "
                "[default=3]\n\t-s Split K slices [int] [default=1]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

  printf("GEMM with dimensions m=%d, n=%d, k=%d\n", m, n, k);

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

  return run(m, n, k, split_k_slices);
}