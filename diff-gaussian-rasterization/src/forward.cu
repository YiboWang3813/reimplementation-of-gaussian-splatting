
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "config.h"
#include "basic_math.cuh"
#include "projection.cuh"
#include "covariance.cuh"
#include "tiling.cuh"
#include "color.cuh"


__global__ void kernel_preprocess(
    int P, int H, int W, 
    dim3 tile_grid, 
    // params 
    const float focal_x, 
    const float focal_y, 
    const float tan_fovx, 
    const float tan_fovy, 
    // gaussian attributes 
    const float* orig_points, 
    const float* scales, 
    const float* quats, 
    const float* opacities, 
    const float* shs, 
    // trans mats 
    const float* viewmatrix, 
    const float* projmatrix, 
    const float* cam_pos, 
    // output 
    int* radii, 
    uint32_t* tile_touched 
)
{
    auto idx = cg::this_grid().thread_rank(); 
    if (idx >= P) return; 

    radii[idx] = 0; 
    tile_touched[idx] = 0; 

    // project and culling 
    const float* p_world = & orig_points[3 * idx]; 
    float p_view[3]; 
    float p_ndc[3]; 
    float p_image[2]; 

    // if the depth in the view space <= z_near, skip this gauss 
    if (!project_and_cull(
        H, W, p_world, viewmatrix, projmatrix, p_view, p_ndc, p_image
    )) return; 

    // compute cov3D always in CUDA 
    const float* scale = &scales[3 * idx]; 
    const float* quat = &quats[4 * idx]; 
    float cov3D[6]; 

    compute_cov3D(scale, quat, cov3D); 

    // compute cov2D 
    float cov2D[3]; 

    compute_cov2D(
        focal_x, focal_y, tan_fovx, tan_fovy, 
        p_world, viewmatrix, cov3D, cov2D 
    ); 

    // get conic (inverse cov2D) under the antialiasing mode 
    float conic[3]; 
    float opacity_scale; 

    // if the det of cov2D <= 0, skip this gauss 
    if (!compute_ewa_conic_and_opacity(
        cov2D, conic, &opacity_scale 
    )) return; 

    // get rect from cov2D 
    float radius = compute_gaussian_radius_from_cov2D(cov2D);  

    uint2 tile_min, tile_max; 
    // if the coverage area is zero, skip this gauss 
    if (!compute_tile_rect_from_pixel_radius(
        p_image, radius, tile_grid, &tile_min, &tile_max 
    )) return; 

    // compute color from sh 
    const float* sh = & shs[idx * MAX_COEFFS * NUM_CHANNELS]; 

    float rgb[3]; 
    bool clamped[3]; 

    compute_color_from_sh(
        SH_DEGREE, MAX_COEFFS, 
        p_world, cam_pos, sh, 
        rgb, clamped
    ); 


}