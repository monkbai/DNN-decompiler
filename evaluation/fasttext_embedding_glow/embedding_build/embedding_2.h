// Bundle API auto-generated header file. Do not edit!
// Glow Tools version: 2021-04-07

#ifndef _GLOW_BUNDLE_EMBEDDING_2_H
#define _GLOW_BUNDLE_EMBEDDING_2_H

#include <stdint.h>

// ---------------------------------------------------------------
//                       Common definitions
// ---------------------------------------------------------------
#ifndef _GLOW_BUNDLE_COMMON_DEFS
#define _GLOW_BUNDLE_COMMON_DEFS

// Glow bundle error code for correct execution.
#define GLOW_SUCCESS 0

// Memory alignment definition with given alignment size
// for static allocation of memory.
#define GLOW_MEM_ALIGN(size)  __attribute__((aligned(size)))

// Macro function to get the absolute address of a
// placeholder using the base address of the mutable
// weight buffer and placeholder offset definition.
#define GLOW_GET_ADDR(mutableBaseAddr, placeholderOff)  (((uint8_t*)(mutableBaseAddr)) + placeholderOff)

#endif

// ---------------------------------------------------------------
//                          Bundle API
// ---------------------------------------------------------------
// Model name: "embedding_2"
// Total data size: 10004800 (bytes)
// Placeholders:
//
//   Name: "input_1"
//   Type: index64<7 x 1>
//   Size: 7 (elements)
//   Size: 56 (bytes)
//   Offset: 0 (bytes)
//
//   Name: "A28"
//   Type: float<1 x 1 x 1>
//   Size: 1 (elements)
//   Size: 4 (bytes)
//   Offset: 64 (bytes)
//
// NOTE: Placeholders are allocated within the "mutableWeight"
// buffer and are identified using an offset relative to base.
// ---------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// Placeholder address offsets within mutable buffer (bytes).
#define EMBEDDING_2_input_1  0
#define EMBEDDING_2_A28      64

// Memory sizes (bytes).
#define EMBEDDING_2_CONSTANT_MEM_SIZE     10001344
#define EMBEDDING_2_MUTABLE_MEM_SIZE      128
#define EMBEDDING_2_ACTIVATIONS_MEM_SIZE  3328

// Memory alignment (bytes).
#define EMBEDDING_2_MEM_ALIGN  64

// Bundle entry point (inference function). Returns 0
// for correct execution or some error code otherwise.
int embedding_2(uint8_t *constantWeight, uint8_t *mutableWeight, uint8_t *activations);

#ifdef __cplusplus
}
#endif
#endif
