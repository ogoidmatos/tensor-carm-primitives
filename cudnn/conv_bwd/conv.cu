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
 *
 * @return A tuple containing the constructed graph, input tensor, filter
 * tensor, and output tensor.
 */
auto build_graph(int n, int c, int h, int w, int k, int r, int s, int padding,
                 int stride, int dilation, int out_h, int out_w,
                 cudnnHandle_t handle, Mode mode) {
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
                                 .set_dim({n, c, h, w})
                                 .set_stride({c * h * w, 1, c * w, c}));

  auto DY =
      graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name("grad")
                        .set_dim({n, k, out_h, out_w})
                        .set_stride({k * out_h * out_w, 1, k * out_w, k}));

  auto conv_options = cudnn_frontend::graph::Conv_wgrad_attributes()
                          .set_padding({padding, padding})
                          .set_stride({stride, stride})
                          .set_dilation({dilation, dilation});

  auto DW = graph->conv_wgrad(DY, input, conv_options);

  DW->set_output(true).set_dim({k, c, r, s});

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

  return std::make_tuple(graph, input, DW, DY);
}

int main(int argc, char *argv[]) {
  int n = 1;
  int c = 3;
  int h = 1024;
  int w = 1024;
  int k = 64;
  int r = 3;
  int s = 3;
  int padding = 0;
  int stride = 1;
  int dilation = 1;
  Mode mode = FP32;

  // parse command line arguments, set args for conv
  int arg;
  cudaSetDevice(0);
  while ((arg = getopt(argc, argv, "n:c:h:w:k:r:s:p:t:d:m:")) != -1)
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
                "[default=1024]\n\t-k \t The number of output channels "
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

  printf("Convolution with settings:\n");
  printf("Batch size: %d\n", n);
  printf("Input channels: %d\n", c);
  printf("Input height: %d\n", h);
  printf("Input width: %d\n", w);
  printf("Output channels: %d\n", k);
  printf("Filter height: %d\n", r);
  printf("Filter width: %d\n", s);
  printf("Padding size: %d\n", padding);
  printf("Stride size: %d\n", stride);
  printf("Dilation size: %d\n", dilation);

  // calculate the output dimensions
  int out_h = (h + 2 * padding - dilation * (r - 1) - 1) / stride + 1;
  int out_w = (w + 2 * padding - dilation * (s - 1) - 1) / stride + 1;

  // allocate memory for input, filter, and output tensors
  float *hostInput = (float *)calloc(n * c * h * w, sizeof(float));
  float *hostFilter = (float *)calloc(k * c * r * s, sizeof(float));
  float *hostOutput = (float *)calloc(n * k * out_h * out_w, sizeof(float));

  // allocate memory for input, filter, and output tensors on device
  float *deviceInput, *deviceFilter, *deviceOutput;
  cudaCheckError(
      cudaMalloc((void **)&deviceInput, n * c * h * w * sizeof(float)));
  cudaCheckError(
      cudaMalloc((void **)&deviceFilter, k * c * r * s * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&deviceOutput,
                            n * k * out_h * out_w * sizeof(float)));

  // copy input and filter tensors to device
  cudaCheckError(cudaMemcpy(deviceInput, hostInput,
                            n * c * h * w * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(deviceOutput, hostOutput,
                            n * k * out_h * out_w * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaDeviceSynchronize());

  cudnnHandle_t handle;
  cudnnCreate(&handle);

  auto [graph, input, DW, DY] =
      build_graph(n, c, h, w, k, r, s, padding, stride, dilation, out_h, out_w,
                  handle, mode);

  int8_t *workspace_ptr;
  cudaCheckError(
      cudaMalloc((void **)&workspace_ptr, graph->get_workspace_size()));

  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>,
                     void *>
      variant_pack = {
          {input, deviceInput}, {DW, deviceFilter}, {DY, deviceOutput}};

  std::cout << *graph << std::endl;

  auto status = graph->execute(handle, variant_pack, workspace_ptr);

  cudaCheckError(cudaDeviceSynchronize());
  std::cout << "Execution status: " << status.get_code() << ":"
            << status.get_message() << std::endl;

  cudaCheckError(cudaMemcpy(hostFilter, deviceFilter,
                            k * c * r * s * sizeof(float),
                            cudaMemcpyDeviceToHost));
  printf("%f\n", hostOutput[0]);

  // free memory
  cudaCheckError(cudaFree(deviceInput));
  cudaCheckError(cudaFree(deviceFilter));
  cudaCheckError(cudaFree(deviceOutput));
  free(hostInput);
  free(hostFilter);
  free(hostOutput);

  cudnnDestroy(handle);
}