/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <vector>

#include "../../../tc-benchmark/nvml_tools.cu"

enum Mode { FP32 = 1, FP16_32 = 2, TF32 = 4 };

// #define POWER

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) \
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
#else
#define cudaCheckError(ans) ans
#endif

// using data_type = cudnn_frontend::DataType_t::FLOAT;

/**
 * Builds a graph for convolution operation using cuDNN.
 *
 * @param n         The batch size.
 * @param c         The number of input channels.
 * @param h         The height of the input tensor.
 * @param w         The width of the input tensor.
 * @param handle    The cuDNN handle.
 *
 * @return A tuple containing the constructed graph, input tensor, filter
 * tensor, and output tensor.
 */
auto build_graph(int n, int c, cudnnHandle_t handle, Mode mode) {
  //   std::thread measuring_thread;
  //   monitor_args thread_args;
  //   thread_args.powerArray = std::vector<int>();
  //   thread_args.clockArray = std::vector<int>();
  //   thread_args.flag = 0;

  //   init_nvml(&thread_args, &measuring_thread);
  //   cudaCheckError(cudaDeviceSynchronize());

  //   thread_args.flag = 1;
  // #ifdef POWER
  // #pragma unroll
  //   for (int i = 0; i < 32768 / 512; i++)
  // #endif

  //   thread_args.flag = 0;
  //   cudaCheckError(&measuring_thread, thread_args.powerArray,
  //                  thread_args.clockArray);

  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
  if (mode & (FP32 | TF32)) {
    graph->set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  } else if (mode & FP16_32) {
    graph->set_io_data_type(cudnn_frontend::DataType_t::HALF)
        .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  }

  auto input = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                 .set_name("input")
                                 .set_dim({n, c, 1})
                                 .set_stride({n, 1, 1}));

  auto amax = graph->reduction(
      input, cudnn_frontend::graph::Reduction_attributes().set_mode(
                 cudnn_frontend::ReductionMode_t::AMAX));

  amax->set_dim({n, 1, 1}).set_stride({1, 1, 1});

  auto asub =
      graph->pointwise(input, amax,
                       cudnn_frontend::graph::Pointwise_attributes().set_mode(
                           cudnn_frontend::PointwiseMode_t::SUB));

  auto aexp = graph->pointwise(
      asub, cudnn_frontend::graph::Pointwise_attributes().set_mode(
                cudnn_frontend::PointwiseMode_t::EXP));

  auto asum = graph->reduction(
      aexp, cudnn_frontend::graph::Reduction_attributes().set_mode(
                cudnn_frontend::ReductionMode_t::ADD));

  asum->set_dim({n, 1, 1}).set_stride({1, 1, 1});

  auto Y =
      graph->pointwise(aexp, asum,
                       cudnn_frontend::graph::Pointwise_attributes().set_mode(
                           cudnn_frontend::PointwiseMode_t::DIV));

  Y->set_output(true);

  graph->validate().is_good();

  graph->build_operation_graph(handle).is_good();

  graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good();

  graph->check_support(handle).is_good();

  auto plan_count = graph->get_execution_plan_count();
  std::cout << "Number of execution plans: " << plan_count << std::endl;

  graph->build_plans(handle).is_good();

  return std::make_tuple(graph, input, Y);
}

int main(int argc, char *argv[]) {
  int n = 1;
  int c = 1024;
  Mode mode = FP32;

  srand(0);

  // parse command line arguments, set args for conv
  int arg;
  cudaSetDevice(0);
  while ((arg = getopt(argc, argv, "n:c:m:")) != -1) switch (arg) {
      case 'n':
        n = atoi(optarg);
        break;
      case 'c':
        c = atoi(optarg);
        break;
      case 'm':
        mode = static_cast<Mode>(atoi(optarg));
        if (mode != FP32 && mode != FP16_32 && mode != TF32) {
          fprintf(stderr, "Invalid mode\n");
          exit(EXIT_FAILURE);
        }
        break;
      default:
        fprintf(stderr,
                "Usage: %s [OPTION]...\n\n\t-n \t The batch size [int] "
                "[default=1]\n\t-c \t The number of input channels [int] "
                "[default=3]\n\t-h \t The height of the input tensor [int] "
                "[default=1024]\n\t-w \t The width of the input tensor [int] "
                "[default=1024]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

  printf("Convolution with settings:\n");
  printf("Batch size: %d\n", n);
  printf("Input channels: %d\n", c);

  // allocate memory for input, filter, and output tensors
  float *hostInput = (float *)calloc(n * c, sizeof(float));
  float *hostOutput = (float *)calloc(n * c, sizeof(float));

  for (int i = 0; i < n * c; i++) {
    hostInput[i] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
  }

  // allocate memory for input, filter, and output tensors on device
  float *deviceInput, *deviceOutput;
  cudaCheckError(cudaMalloc((void **)&deviceInput, n * c * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&deviceOutput, n * c * sizeof(float)));

  // copy input and filter tensors to device
  cudaCheckError(cudaMemcpy(deviceInput, hostInput, n * c * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaDeviceSynchronize());

  cudnnHandle_t handle;
  cudnnCreate(&handle);

  auto [graph, input, Y] = build_graph(n, c, handle, mode);

  int8_t *workspace_ptr;
  cudaCheckError(
      cudaMalloc((void **)&workspace_ptr, graph->get_workspace_size()));

  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>,
                     void *>
      variant_pack = {{input, deviceInput}, {Y, deviceOutput}};

  std::cout << *graph << std::endl;

  auto status = graph->execute(handle, variant_pack, workspace_ptr);

  cudaCheckError(cudaDeviceSynchronize());
  std::cout << "Execution status: " << status.get_code() << ":"
            << status.get_message() << std::endl;

  cudaCheckError(cudaMemcpy(hostOutput, deviceOutput, n * c * sizeof(float),
                            cudaMemcpyDeviceToHost));

  printf("%f\n", hostOutput[0]);

  std::cout << "Elements processed: " << n * c << std::endl;

  // free memory
  cudaCheckError(cudaFree(deviceInput));
  cudaCheckError(cudaFree(deviceOutput));
  free(hostInput);
  free(hostOutput);

  cudnnDestroy(handle);
}