#pragma once
#include <cuda_runtime.h>


__device__ __forceinline__ 
void mat4_mul_vec4_rowmajor(
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


__device__ __forceinline__
void mat3_mul_mat3_rowmajor(
    const float* A, // [9]
    const float* B, // [9]
    float* C // [9]
)
{
    #pragma unroll 
    for (int i = 0; i < 3; i++)
    {
        float ai0 = A[3 * i + 0], ai1 = A[3 * i + 1], ai2 = A[3 * i + 2]; 
        C[i * 3 + 0] = ai0 * B[0] + ai1 * B[3] + ai2 * B[6]; 
        C[i * 3 + 1] = ai0 * B[1] + ai1 * B[4] + ai2 * B[7]; 
        C[i * 3 + 2] = ai0 * B[2] + ai1 * B[5] + ai2 * B[8]; 
    }
}


__device__ __forceinline__ 
void mat3_transpose_rowmajor(
    const float* A, // [9]
    float* AT // [9]
)
{
    AT[0] = A[0]; AT[1] = A[3]; AT[2] = A[6]; 
    AT[3] = A[1]; AT[4] = A[4]; AT[5] = A[7]; 
    AT[6] = A[2]; AT[7] = A[5]; AT[8] = A[8]; 
}