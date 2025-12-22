#pragma once
#include "basic_math.cuh"


__device__ __forceinline__
bool in_frustum(
    int idx,
    const float* orig_points,   // [P,3]
    const float* viewmatrix,    // [16] 
    const float* projmatrix,    // [16] 
    bool prefiltered,
    float* p_view,          // [3]
    float near_z = 0.2f
)
{
    // world space 
    float p_world[3]; 
    p_world[0] = orig_points[3 * idx + 0]; 
    p_world[1] = orig_points[3 * idx + 1]; 
    p_world[2] = orig_points[3 * idx + 2]; 

    float p_world_tilde[4]; 
    p_world_tilde[0] = p_world[0];
    p_world_tilde[1] = p_world[1];
    p_world_tilde[2] = p_world[2]; 
    p_world_tilde[3] = 1.0f; 

    // view space 
    float p_view_tilde[4]; 
    mat4_mul_vec4(viewmatrix, p_world_tilde, p_view_tilde); 

    p_view[0] = p_view_tilde[0]; 
    p_view[1] = p_view_tilde[1]; 
    p_view[2] = p_view_tilde[2]; 

    // clip space 
    float p_clip[4]; 
    mat4_mul_vec4(projmatrix, p_view_tilde, p_clip); 

    float inv_w = 1.0f / (p_clip[3] + 1e-7f); 
    float p_ndc[3]; 
    p_ndc[0] = p_clip[0] * inv_w; 
    p_ndc[1] = p_clip[1] * inv_w; 
    p_ndc[2] = p_clip[2] * inv_w; 

    // near place cull 
    if (p_view[2] <= near_z /* ||
        p_ndc[0] < -1.3f || p_ndc[0] > 1.3f ||
        p_ndc[1] < -1.3f || p_ndc[1] > 1.3f */) 
    {
        if (prefiltered) {
            printf("Point is filtered although prefiltered is set. This shouldn't happen!\n");
            __trap();
        }
        return false;
    }
    return true; 
}