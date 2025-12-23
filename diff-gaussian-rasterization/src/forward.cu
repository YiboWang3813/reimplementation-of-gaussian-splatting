
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "basic_math.cuh"
#include "projection.cuh"
#include "covariance.cuh"


__global__ void kernel_preprocess(
    int n_gauss, 
    const float* orig_points, 
    const float* scales, 
    const float* quats, 
    const float* viewmatrix, 
    const float* projmatrix, 
    // output 
    int* radii, 
    uint32_t* tile_touched, 
    bool prefiltered
)
{
    auto idx = cg::this_grid().thread_rank(); 
    if (idx >= n_gauss) return; 

    radii[idx] = 0; 
    tile_touched[idx] = 0; 

    // near culling 
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered)) 
        return; 

    // transform point 
    float p_world_tilde[4] = {
        orig_points[3 * idx + 0], 
        orig_points[3 * idx + 1], 
        orig_points[3 * idx + 2], 
        1.0f 
    }; 
    float p_view_tilde[4]; 
    mat4_mul_vec4_rowmajor(viewmatrix, p_world_tilde, p_view_tilde); 
    float p_clip[4]; 
    mat4_mul_vec4_rowmajor(projmatrix, p_view_tilde, p_clip); 
    float inv_w = 1.0f / (p_clip[3] + 1e-6f); 
    float p_ndc[3] = {
        p_clip[0] * inv_w, 
        p_clip[1] * inv_w, 
        p_clip[2] * inv_w 
    };

    // compute cov3D always in CUDA 
    float cov3D[6]; 
    compute_cov3D(idx, scales, quats, cov3D); 



    
}