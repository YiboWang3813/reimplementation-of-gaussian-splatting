
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "basic_math.cuh"
#include "projection.cuh"
#include "covariance.cuh"


__global__ void kernel_preprocess(
    int n_gauss, 
    // params 
    const float focal_x, 
    const float focal_y, 
    const float tan_fovx, 
    const float tan_fovy, 
    // gaussian attributes 
    const float* orig_points, 
    const float* scales, 
    const float* quats, 
    const float* viewmatrix, 
    const float* projmatrix, 
    // output 
    int* radii, 
    uint32_t* tile_touched, 
    // control 
    bool prefiltered, 
    bool antialiasing
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
    float p_ndc[3] = {p_clip[0] * inv_w, p_clip[1] * inv_w, p_clip[2] * inv_w};

    // compute cov3D always in CUDA 
    const float* scale = &scales[3 * idx]; 
    const float* quat = &quats[4 * idx]; 
    float cov3D[6]; 
    compute_cov3D(scale, quat, cov3D); 

    // compute cov2D 
    float cov2D[3]; 
    compute_cov2D(
        focal_x, focal_y, tan_fovx, tan_fovy, 
        p_world_tilde, viewmatrix, cov3D, cov2D 
    ); 

    // get conic (inverse cov2D) under the antialiasing mode 
    float opacity_scale = 1.0f; 

    float cov2D_filtered[3] = {cov2D[0], cov2D[1], cov2D[2]}; 

    if (antialiasing)
    {
        const float h_var = 0.3f; 
        cov2D_filtered[0] += h_var; 
        cov2D_filtered[2] += h_var; 

        float det_cov2D = cov2D[0] * cov2D[2] 
                        - cov2D[1] * cov2D[1]; 
        float det_cov2D_filtered = cov2D_filtered[0] * cov2D_filtered[2] 
                                - cov2D_filtered[1] * cov2D_filtered[1];

        opacity_scale = sqrt(max(2.5e-5f, det_cov2D / det_cov2D_filtered)); 
    }

    float conic[3]; 
    float det_cov2D_filtered = cov2D_filtered[0] * cov2D_filtered[2] 
                            - cov2D_filtered[1] * cov2D_filtered[1];
    float inv_det_filtered = 1.0f / det_cov2D_filtered; 
    conic[0] = inv_det_filtered * cov2D_filtered[2]; 
    conic[1] = -inv_det_filtered * cov2D_filtered[1]; 
    conic[2] = inv_det_filtered * cov2D_filtered[0]; 

    
}