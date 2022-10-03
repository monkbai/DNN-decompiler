/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <tvm/runtime/c_runtime_api.h>

#include "bundle.h"

extern const char build_graph_c_json[];
extern unsigned int build_graph_c_json_len;

extern const char build_params_c_bin[];
extern unsigned int build_params_c_bin_len;

#define OUTPUT_LEN 1

int main(int argc, char** argv) {
  //assert(argc == 2 && "Usage: demo_static <cat.bin>");

  char* json_data = (char*)(build_graph_c_json);
  char* params_data = (char*)(build_params_c_bin);
  uint64_t params_size = build_params_c_bin_len;

  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);

  void* handle = tvm_runtime_create(json_data, params_data, params_size, argv[0]);
  gettimeofday(&t1, 0);

  //float input_storage[1 * 3 * 224 * 224];
  //FILE* fp = fopen(argv[1], "rb");
  //(void)fread(input_storage, 3 * 224 * 224, 4, fp);
  //fclose(fp);
  
  // input 1
  int64_t input_storage[7];
  for (int i=1; i<=7; i++){
    input_storage[i-1] = atoi(argv[i]);
  }
  /* "This film is terrible"
  input_storage[0] = 70;
  input_storage[1] = 24;
  input_storage[2] = 9;
  input_storage[3] = 676;
  input_storage[4] = 285;
  input_storage[5] = 816;
  input_storage[6] = 6514;
  */
  // "This film is great"
  /*input_storage[0] = 70;
  input_storage[1] = 24;
  input_storage[2] = 9;
  input_storage[3] = 113;
  input_storage[4] = 285;
  input_storage[5] = 816;
  input_storage[6] = 2382;
  */
  
  DLTensor input;
  input.data = input_storage;
  DLContext ctx = {kDLCPU, 0};
  input.ctx = ctx;
  input.ndim = 2;
  // DLDataType dtype = {kDLFloat, 32, 1};
  DLDataType dtype = {kDLInt, 64, 1};
  input.dtype = dtype;
  int64_t shape[2] = {7, 1};
  input.shape = shape;
  input.strides = NULL;
  input.byte_offset = 0;

  tvm_runtime_set_input(handle, "input.1", &input);
  
  
  gettimeofday(&t2, 0);
  
  tvm_runtime_run(handle);
  
  gettimeofday(&t3, 0);

  float output_storage[1];
  DLTensor output;
  output.data = output_storage;
  DLContext out_ctx = {kDLCPU, 0};
  output.ctx = out_ctx;
  output.ndim = 3;
  DLDataType out_dtype = {kDLFloat, 32, 1};
  output.dtype = out_dtype;
  int64_t out_shape[3] = {1, 1, 1};
  output.shape = out_shape;
  output.strides = NULL;
  output.byte_offset = 0;

  
  tvm_runtime_get_output(handle, 0, &output);
  
  gettimeofday(&t4, 0);


  for (int i = 0; i < OUTPUT_LEN; ++i) {
    printf("%f, ", output_storage[i]);
  }
  printf("\n");

  tvm_runtime_destroy(handle);
  gettimeofday(&t5, 0);

  printf(
      "timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
      "%.2f ms (get_output), %.2f ms (destroy)\n",
      (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000.f,
      (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.f,
      (t3.tv_sec - t2.tv_sec) * 1000 + (t3.tv_usec - t2.tv_usec) / 1000.f,
      (t4.tv_sec - t3.tv_sec) * 1000 + (t4.tv_usec - t3.tv_usec) / 1000.f,
      (t5.tv_sec - t4.tv_sec) * 1000 + (t5.tv_usec - t4.tv_usec) / 1000.f);

  return 0;
}
