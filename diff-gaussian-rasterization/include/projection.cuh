// include/projection.cuh 
#pragma once
#include "basic_math.cuh"
#include "config.h"


__device__ __forceinline__
int ndc2Pix(float v, int S)
{
	return (int) (((v + 1.0) * S - 1.0) * 0.5); 
}


__device__ __forceinline__
bool project_and_cull(
    int H, int W, 
    const float* p_world, // [3] 
    const float* viewmatrix,    // [16]
    const float* projmatrix,    // [16]
    float* p_view, // [3] 
    float* p_ndc, // [3] 
    int* p_image // [2] 
)
{
    // world space 
    float p_world_tilde[4] = {
        p_world[0], p_world[1], p_world[2], 1.0f
    }; 

    // view space 
    float p_view_tilde[4];
    mat4_mul_vec4_rowmajor(viewmatrix, p_world_tilde, p_view_tilde);

    p_view[0] = p_view_tilde[0];
    p_view[1] = p_view_tilde[1];
    p_view[2] = p_view_tilde[2];

    // near place culling (view space) 
    if (p_view[2] <= near_z)
    {
        if (prefiltered)
        {
            printf("Point is filtered although prefiltered is set. This shouldn't happen!\n");
            __trap();
        }
        return false;
    }

    // clip space 
    float p_clip[4];
    mat4_mul_vec4_rowmajor(projmatrix, p_view_tilde, p_clip);

    float inv_w = 1.0f / (p_clip[3] + 1e-6f);

    // NDC space 
    p_ndc[0] = p_clip[0] * inv_w;
    p_ndc[1] = p_clip[1] * inv_w;
    p_ndc[2] = p_clip[2] * inv_w;

    // pixel space 
    p_image[0] = ndc2Pix(p_ndc[0], W); 
    p_image[1] = ndc2Pix(p_ndc[1], H); 

    return true; 
}
