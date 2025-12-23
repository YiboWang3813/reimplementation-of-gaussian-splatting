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


__device__ __forceinline__
void compute_cov2D(
    float focal_x, float focal_y, 
    float tan_fovx, float tan_fovy, 
    const float* p_world_tilde, // [4]
    const float* viewmatrix, // [16] 
    const float* cov3D, // [6] 
    float* cov2D // [3]
)
{
    // compute p_view_tilde 
    float p_view_tile[4]; 
    mat4_mul_vec4_rowmajor(viewmatrix, p_world_tilde, p_view_tile); 

    // constrain x and y component of p_view 
    float limx = 1.3f * tan_fovx; 
    float limy = 1.3f * tan_fovy; 
    float xv_zv = p_view_tile[0] / p_view_tile[2]; 
    float yv_zv = p_view_tile[1] / p_view_tile[2]; 
    p_view_tile[0] = fminf(limx, fmaxf(-limx, xv_zv)) * p_view_tile[2]; 
    p_view_tile[1] = fminf(limy, fmaxf(-limy, yv_zv)) * p_view_tile[2]; 
    
    // compute Jacobian J 
    float J[9] = {0.0f}; 
    J[0] = focal_x / p_view_tile[2]; 
    J[2] = - (focal_x * p_view_tile[0]) / (p_view_tile[2] * p_view_tile[2]); 
    J[4] = focal_y / p_view_tile[2]; 
    J[5] = - (focal_y * p_view_tile[1]) / (p_view_tile[2] * p_view_tile[2]); 

    // get rotation part of viewmatrix 
    float Rview[9]; 
    Rview[0] = viewmatrix[0]; Rview[1] = viewmatrix[1]; Rview[2] = viewmatrix[2]; 
    Rview[3] = viewmatrix[4]; Rview[4] = viewmatrix[5]; Rview[5] = viewmatrix[6]; 
    Rview[6] = viewmatrix[8]; Rview[7] = viewmatrix[9]; Rview[8] = viewmatrix[10]; 

    // compute T = J * Rview 
    float T[9]; 
    mat3_mul_mat3_rowmajor(J, Rview, T); 

    // complete full Sigma using cov3D 
    float Sigma[9]; 
    Sigma[0] = cov3D[0]; Sigma[1] = cov3D[1]; Sigma[2] = cov3D[2]; 
    Sigma[3] = cov3D[1]; Sigma[4] = cov3D[3]; Sigma[5] = cov3D[4]; 
    Sigma[6] = cov3D[2]; Sigma[7] = cov3D[4]; Sigma[8] = cov3D[5]; 

    // compute Sigma2D = T * Sigma3D * T^T 
    float Sigma2D[9]; 
    float tmp[9]; 
    mat3_mul_mat3_rowmajor(T, Sigma, tmp);
    float T_trans[9]; 
    mat3_transpose_rowmajor(T, T_trans); 
    mat3_mul_mat3_rowmajor(tmp, T_trans, Sigma2D); 

    // save 2D part of Sigma2D as cov2D 
    cov2D[0] = Sigma2D[0]; 
    cov2D[1] = Sigma2D[1]; 
    cov2D[2] = Sigma2D[4]; 
}