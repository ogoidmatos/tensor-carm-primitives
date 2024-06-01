/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <vector>

#include "cublas_utils.h"

using data_type = float;

int run(int m, int n, int k) {
  const int length_m = m;
  const int length_n = n;
  const int length_k = k;

  const std::vector<data_type> A(length_m * length_k, 1.0);
  const std::vector<data_type> B(length_k * length_n, 1.0);
  std::vector<data_type> C(length_m * length_n, 0.0);

  const data_type alpha = 1.0;
  const data_type beta = 0.0;

  data_type *d_A = nullptr;
  data_type *d_B = nullptr;
  data_type *d_C = nullptr;

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // Set math mode to allow TF32 tensor core operations
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  CUDA_CHECK(cudaMalloc((void **)&d_A, sizeof(data_type) * A.size()));
  CUDA_CHECK(cudaMalloc((void **)&d_B, sizeof(data_type) * B.size()));
  CUDA_CHECK(cudaMalloc((void **)&d_C, sizeof(data_type) * C.size()));

  CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(data_type) * A.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(data_type) * B.size(),
                        cudaMemcpyHostToDevice));

  CUBLAS_CHECK(cublasSgemm(handle, transa, transb, length_m, length_n, length_k,
                           &alpha, d_A, length_k, d_B, length_k, &beta, d_C,
                           length_n));

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(C.data(), d_C, sizeof(data_type) * C.size(),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUBLAS_CHECK(cublasDestroy(handle));

  CUDA_CHECK(cudaDeviceReset());

  printf("C[0] = %f\n", C[0]);

  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  int m = 4096;
  int n = 4096;
  int k = 4096;

  int c;
  cudaSetDevice(0);
  while ((c = getopt(argc, argv, "m:n:k:a:h")) != -1) switch (c) {
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
      case 'h':
        fprintf(stdout,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=4096]\n\t-n \t N "
                "dimension [int] [default=4096]\n\t-k \t K dimension [int] "
                "[default=4096]\n\t-a \t All "
                "dimensions [int]\n\n",
                argv[0]);
        exit(EXIT_SUCCESS);
      default:
        fprintf(stderr,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=4096]\n\t-n \t N "
                "dimension [int] [default=4096]\n\t-k \t K dimension [int] "
                "[default=4096]\n\t-a \t All "
                "dimensions [int]\n\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

  printf("GEMM with dimensions m=%d, n=%d, k=%d\n", m, n, k);

  return run(m, n, k);
}