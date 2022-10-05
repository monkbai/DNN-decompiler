/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <assert.h>
#include <inttypes.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>

#include "embedding_2.h"



//===----------------------------------------------------------------------===//
//                 Wrapper code for executing a bundle
//===----------------------------------------------------------------------===//
/// Statically allocate memory for constant weights (model weights) and
/// initialize.
GLOW_MEM_ALIGN(EMBEDDING_2_MEM_ALIGN)
uint8_t constantWeight[EMBEDDING_2_CONSTANT_MEM_SIZE] = {
#include "embedding_2.weights.txt"
};

/// Statically allocate memory for mutable weights (model input/output data).
GLOW_MEM_ALIGN(EMBEDDING_2_MEM_ALIGN)
uint8_t mutableWeight[EMBEDDING_2_MUTABLE_MEM_SIZE];

/// Statically allocate memory for activations (model intermediate results).
GLOW_MEM_ALIGN(EMBEDDING_2_MEM_ALIGN)
uint8_t activations[EMBEDDING_2_ACTIVATIONS_MEM_SIZE];

/// Bundle input data absolute address.
uint8_t *inputAddr = GLOW_GET_ADDR(mutableWeight, EMBEDDING_2_input_1);

/// Bundle output data absolute address.
uint8_t *outputAddr = GLOW_GET_ADDR(mutableWeight, EMBEDDING_2_A28);


/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
static void printResults() {
  float *results = (float *)(outputAddr);
  for (int i = 0; i < 1; ++i) {
    printf("%f, ", results[i]);
    
  }
  printf("\n");
}

int main(int argc, char **argv) {
  //parseCommandLineOptions(argc, argv);

  // Initialize input images.
  //initInputImages();
  int64_t *input = inputAddr;
  for (int i=1; i<=7; i++){
    *(input+i-1) = atoi(argv[i]);
  }
  //*(int64_t *)inputAddr = 7;
  //*(input+0) = 70;
  //*(input+1) = 24;
  //*(input+2) = 9;
  //*(input+3) = 113;
  //*(input+4) = 285;
  //*(input+5) = 816;
  //*(input+6) = 2382;
  
  // Perform the computation.
  int errCode = embedding_2(constantWeight, mutableWeight, activations);
  if (errCode != GLOW_SUCCESS) {
    printf("Error running bundle: error code %d\n", errCode);
  }

  // Print results.
  printResults();
}