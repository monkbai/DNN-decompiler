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

#include "resnet18_v1_7.h"

/// This is an example demonstrating how to use auto-generated bundles and
/// create standalone executables that can perform neural network computations.
/// This example loads and runs the compiled lenet_mnist network model.
/// This example is using the static bundle API.

#define DEFAULT_HEIGHT 224
#define DEFAULT_WIDTH 224
#define OUTPUT_LEN 1000


//===----------------------------------------------------------------------===//
//                 Wrapper code for executing a bundle
//===----------------------------------------------------------------------===//
/// Statically allocate memory for constant weights (model weights) and
/// initialize.
GLOW_MEM_ALIGN(RESNET18_V1_7_MEM_ALIGN)
uint8_t constantWeight[RESNET18_V1_7_CONSTANT_MEM_SIZE] = {
#include "resnet18_v1_7.weights.txt"
};

/// Statically allocate memory for mutable weights (model input/output data).
GLOW_MEM_ALIGN(RESNET18_V1_7_MEM_ALIGN)
uint8_t mutableWeight[RESNET18_V1_7_MUTABLE_MEM_SIZE];

/// Statically allocate memory for activations (model intermediate results).
GLOW_MEM_ALIGN(RESNET18_V1_7_MEM_ALIGN)
uint8_t activations[RESNET18_V1_7_ACTIVATIONS_MEM_SIZE];

/// Bundle input data absolute address.
uint8_t *inputAddr = GLOW_GET_ADDR(mutableWeight, RESNET18_V1_7_data);

/// Bundle output data absolute address.
uint8_t *outputAddr = GLOW_GET_ADDR(mutableWeight, RESNET18_V1_7_resnetv15_dense0_fwd__1);


/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
static void printResults() {
  int maxIdx = 0;
  float maxValue = 0;
  float *results = (float *)(outputAddr);
  for (int i = 0; i < OUTPUT_LEN; ++i) {
    if (results[i] > maxValue) {
      maxValue = results[i];
      maxIdx = i;
    }
  }
  printf("Result: %u\n", maxIdx);
  printf("Confidence: %f\n", maxValue);
}

int main(int argc, char **argv) {
  //parseCommandLineOptions(argc, argv);

  // Initialize input images.
  //initInputImages();
  if (argc == 2){
    FILE *fid = NULL;
    fid = fopen(argv[1], "rb");
    fread(inputAddr, sizeof(char), RESNET18_V1_7_resnetv15_dense0_fwd__1, fid);
    printf("read %d bytes from %s\n", RESNET18_V1_7_resnetv15_dense0_fwd__1, argv[1]);
    fclose(fid);
  }
  else{
    printf("input file not provided\nexit\n");
  }
  // Perform the computation.
  int errCode = resnet18_v1_7(constantWeight, mutableWeight, activations);
  if (errCode != GLOW_SUCCESS) {
    printf("Error running bundle: error code %d\n", errCode);
  }

  // Print results.
  printResults();
}