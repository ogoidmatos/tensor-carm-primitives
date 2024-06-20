#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

#include "../../../tc-benchmark/nvml_tools.cu"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
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
using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

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

// Split K dimension into 1 partitions
constexpr int split_k_slices = 1;

// Which iterator algorithm to use: Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
    cutlass::conv::IteratorAlgorithm::kOptimized;

// Is the output packed or strided
// Use kStride if using strided output
static cutlass::conv::StrideSupport const OutputStride =
    cutlass::conv::StrideSupport::kUnity;

// Kernel properties type
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages,
    cutlass::arch::OpMultiplyAdd, IteratorAlgorithm, OutputStride>::Kernel;

// Type of the actual kernel
using ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

/**
 * Performs convolution using cutlass.
 *
 * @param n         The batch size.
 * @param c         The number of input channels.
 * @param h         The height of the input tensor.
 * @param w         The width of the input tensor.
 * @param k         The number of output channels (filters).
 * @param r         The height of the filter.
 * @param s         The width of the filter.
 * @param padding   The padding size.
 * @param stride    The stride size.
 * @param dilation  The dilation size.
 * @param handle    The cuDNN handle.
 *.
 */
int run(int n, int c, int h, int w, int k, int r, int s, int padding,
        int stride, int dilation) {
  std::thread measuring_thread;
  monitor_args thread_args;
  thread_args.powerArray = std::vector<int>();
  thread_args.clockArray = std::vector<int>();
  thread_args.flag = 0;

  init_nvml(&thread_args, &measuring_thread);
  cudaDeviceSynchronize();

  int out_h = (h + 2 * padding - dilation * (r - 1) - 1) / stride + 1;
  int out_w = (w + 2 * padding - dilation * (s - 1) - 1) / stride + 1;

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      cutlass::Tensor4DCoord(n, h, w, c));
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      cutlass::Tensor4DCoord(k, r, s, c));
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      cutlass::Tensor4DCoord(n, out_h, out_w, k));
  printf("Tensors initialized\n");
  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFill(tensor_a.host_view());
  cutlass::reference::host::TensorFill(tensor_b.host_view());
  cutlass::reference::host::TensorFill(tensor_c.host_view());

  printf("Tensors filled\n");
  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  printf("Data copied to device\n");
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  cutlass::conv::Conv2dProblemSize problem_size(
      cutlass::Tensor4DCoord(n, h, w, c), cutlass::Tensor4DCoord(k, r, s, c),
      cutlass::Tensor4DCoord(padding, padding, padding, padding),
      cutlass::MatrixCoord(stride, stride),
      cutlass::MatrixCoord(dilation, dilation),
      cutlass::Tensor4DCoord(n, out_h, out_w, k), mode, split_k_slices);

  typename ImplicitGemm::Arguments arguments{
      problem_size,           // <- problem size of matrix multiplication
      tensor_a.device_ref(),  // <- reference to matrix A on device
      tensor_b.device_ref(),  // <- reference to matrix B on device
      tensor_c.device_ref(),  // <- reference to matrix C on device
      tensor_c.device_ref(),  // <- reference to matrix D on device
      {alpha, beta}};         // <- tuple of alpha and beta

  // Instantiate CUTLASS kernel depending on templates
  ImplicitGemm conv_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = conv_op.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = conv_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  printf("Problem size is supported\n");
  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = conv_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  printf("Initialized\n");
  thread_args.flag = 1;
#ifdef POWER
#pragma unroll
  for (int i = 0; i < 32768 * 64; i++)
#endif
    // Launch initialized CUTLASS kernel
    status = conv_op();
  cudaDeviceSynchronize();
  thread_args.flag = 0;
  stop_nvml(&measuring_thread, thread_args.powerArray, thread_args.clockArray);

  CUTLASS_CHECK(status);
  printf("Launched\n");

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_c.sync_host();

  return tensor_c.host_ref().at({0, 0, 0, 0});
}

int main(int argc, char *argv[]) {
  int n = 1;
  int c = 3;
  int h = 256;
  int w = 256;
  int k = 64;
  int r = 3;
  int s = 3;
  int padding = 0;
  int stride = 1;
  int dilation = 1;

  // parse command line arguments, set args for conv
  int arg;
  cudaSetDevice(0);
  while ((arg = getopt(argc, argv, "n:c:h:w:k:r:s:p:t:d:")) != -1)
    switch (arg) {
      case 'n':
        n = atoi(optarg);
        break;
      case 'c':
        c = atoi(optarg);
        break;
      case 'h':
        h = atoi(optarg);
        break;
      case 'w':
        w = atoi(optarg);
        break;
      case 'k':
        k = atoi(optarg);
        break;
      case 'r':
        r = atoi(optarg);
        break;
      case 's':
        s = atoi(optarg);
        break;
      case 'p':
        padding = atoi(optarg);
        break;
      case 't':
        stride = atoi(optarg);
        break;
      case 'd':
        dilation = atoi(optarg);
        break;
      default:
        fprintf(stderr,
                "Usage: %s [OPTION]...\n\n\t-n \t The batch size [int] "
                "[default=1]\n\t-c \t The number of input channels [int] "
                "[default=3]\n\t-h \t The height of the input tensor [int] "
                "[default=256]\n\t-w \t The width of the input tensor [int] "
                "[default=256]\n\t-k \t The number of output channels "
                "(filters) [int] "
                "[default=64]\n\t-r \t The height of the filter [int] "
                "[default=3]\n\t-s \t The width of the filter [int] "
                "[default=3]\n\t-p \t The padding size [int] "
                "[default=0]\n\t-t \t The stride size [int] "
                "[default=1]\n\t-d \t The dilation size [int] "
                "[default=1]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

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

  return run(n, c, h, w, k, r, s, padding, stride, dilation);
}