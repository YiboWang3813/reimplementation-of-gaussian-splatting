#pragma once 
#include "basic_math.cuh"


__device__ __forceinline__ 
void quat_to_rotmat_rowmajor(
    const float* q, // [4] r,x,y,z 
    float* R // [9]
)
{
    float r = q[0], x = q[1], y = q[2], z = q[3]; 

    R[0] = 1.f - 2.f * (y * y + z * z); 
    R[1] = 2.f * (x * y - r * z); 
    R[2] = 2.f * (x * z + r * y); 

    R[3] = 2.f * (x * y + r * z); 
    R[4] = 1.f - 2.f * (x * x + z * z); 
    R[5] = 2.f * (y * z - r * x); 

    R[6] = 2.f * (x * z - r * y); 
    R[7] = 2.f * (y * z + r * x); 
    R[8] = 1.f - 2.f * (x * x + y * y); 
}


__device__ __forceinline__
void compute_cov3D(
    const float* scale, // [3]
    const float* quat, // [4] r,x,y,z
    float* cov3D // [6] [xx, xy, xz, yy, yz, zz]
)
{
    float S[9] = {0.0f}; 
    S[0] = scale[0]; 
    S[4] = scale[1]; 
    S[8] = scale[2];  
    
    float R[9]; 
    quat_to_rotmat_rowmajor(quat, R); 

    float M[9]; 
    mat3_mul_mat3_rowmajor(S, R, M); 

    float MT[9]; 
    mat3_transpose_rowmajor(M, MT); 

    float Sigma[9]; 
    mat3_mul_mat3_rowmajor(MT, M, Sigma); 

    cov3D[0] = Sigma[0]; 
    cov3D[1] = Sigma[1]; 
    cov3D[2] = Sigma[2]; 
    cov3D[3] = Sigma[4];
    cov3D[4] = Sigma[5];
    cov3D[5] = Sigma[8];
}