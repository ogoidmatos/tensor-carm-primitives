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

// #define POWER

enum Mode { FP32 = 1, FP16_32 = 2, TF32 = 4 };

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
 * @param inputs    The number of input channels.
 * @param outputs   The number of output channels.
 * @param handle    The cuDNN handle.
 *
 * @return A tuple containing the constructed graph, input tensor, filter
 * tensor, and output tensor.
 */
auto build_graph(int n, int inputs, int outputs, cudnnHandle_t handle,
                 Mode mode) {
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
                                 .set_dim({n, inputs})
                                 .set_stride({n, 1}));

  auto W = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("weights")
                             .set_dim({inputs, outputs})
                             .set_stride({inputs, 1}));

  auto matmul_options = cudnn_frontend::graph::Matmul_attributes()
                            .set_padding({padding, padding})
                            .set_stride({stride, stride})
                            .set_dilation({dilation, dilation});

  auto Y = graph->conv_fprop(input, W, conv_options);

  Y->set_output(true);

  graph->validate().is_good();

  graph->build_operation_graph(handle).is_good();

  graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good();

  if (mode & (FP16_32 | TF32)) {
    graph->select_numeric_notes(std::vector<cudnn_frontend::NumericalNote_t>(
        1, cudnn_frontend::NumericalNote_t::TENSOR_CORE));
  }

  graph->check_support(handle).is_good();

  auto plan_count = graph->get_execution_plan_count();
  std::cout << "Number of execution plans: " << plan_count << std::endl;

  graph->build_plans(handle).is_good();

  return std::make_tuple(graph, input, W, Y);
}

int main(int argc, char *argv[]) {
  int n = 1;
  int inputs = 1024;
  int outputs = 1024;
  Mode mode = FP32;

  // parse command line arguments, set args for conv
  int arg;
  cudaSetDevice(0);
  while ((arg = getopt(argc, argv, "n:i:o:m:")) != -1) switch (arg) {
      case 'n':
        n = atoi(optarg);
        break;
      case 'i':
        inputs = atoi(optarg);
        break;
      case 'o':
        outputs = atoi(optarg);
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
                "[default=1]\n\t-i \t The number of input neurons [int] "
                "[default=1024]\n\t-o \t The number of output neurons "
                "[int] "
                "[default=1024]\n\t-m \t The mode [int] "
                "[default=1]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

  printf("Fully connected layer with settings:\n");
  printf("\tBatch size: %d\n", n);
  printf("\tNumber of input neurons: %d\n", inputs);
  printf("\tNumber of output neurons: %d\n", outputs);

  // allocate memory for input, weights, and output tensors
  float *hostInput = (float *)calloc(n * inputs, sizeof(float));
  float *hostWeights = (float *)calloc(inputs * outputs, sizeof(float));
  float *hostOutput = (float *)calloc(n * outputs, sizeof(float));

  // allocate memory for input, filter, and output tensors on device
  float *deviceInput, *deviceWeights, *deviceOutput;
  cudaCheckError(cudaMalloc((void **)&deviceInput, n * inputs * sizeof(float)));
  cudaCheckError(
      cudaMalloc((void **)&deviceWeights, inputs * outputs * sizeof(float)));
  cudaCheckError(
      cudaMalloc((void **)&deviceOutput, n * outputs * sizeof(float)));

  // copy input and filter tensors to device
  cudaCheckError(cudaMemcpy(deviceInput, hostInput, n * inputs * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(deviceWeights, hostWeights,
                            inputs * outputs * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaDeviceSynchronize());

  cudnnHandle_t handle;
  cudnnCreate(&handle);

  auto [graph, input, W, Y] = build_graph(n, inputs, outputs, handle, mode);

  int8_t *workspace_ptr;
  cudaCheckError(
      cudaMalloc((void **)&workspace_ptr, graph->get_workspace_size()));

  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>,
                     void *>
      variant_pack = {
          {input, deviceInput}, {W, deviceWeights}, {Y, deviceOutput}};

  std::cout << *graph << std::endl;

  auto status = graph->execute(handle, variant_pack, workspace_ptr);

  cudaCheckError(cudaDeviceSynchronize());
  std::cout << "Execution status: " << status.get_code() << ":"
            << status.get_message() << std::endl;

  cudaCheckError(cudaMemcpy(hostOutput, deviceOutput,
                            n * k * out_h * out_w * sizeof(float),
                            cudaMemcpyDeviceToHost));
  printf("%f\n", hostOutput[0]);

  // free memory
  cudaCheckError(cudaFree(deviceInput));
  cudaCheckError(cudaFree(deviceWeights));
  cudaCheckError(cudaFree(deviceOutput));
  free(hostInput);
  free(hostWeights);
  free(hostOutput);

  cudnnDestroy(handle);
}