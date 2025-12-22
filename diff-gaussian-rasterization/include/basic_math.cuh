#pragma once
#include <cuda_runtime.h>


__device__ __forceinline__ 
void mat4_mul_vec4(
    const float* M, // [16]
    const float* p, // [4]
    float* out // [4]
)
{
    out[0] = M[0] * p[0] + M[1] * p[1] + M[2] * p[2] + M[3] * p[3]; 
    out[1] = M[4] * p[0] + M[5] * p[1] + M[6] * p[2] + M[7] * p[3]; 
    out[2] = M[8] * p[0] + M[9] * p[1] + M[10] * p[2] + M[11] * p[3]; 
    out[3] = M[12] * p[0] + M[13] * p[1] + M[14] * p[2] + M[15] * p[3]; 
}