
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
    const float* orig_points, // [P, 3]
    const float* scales, // [P, 3]
    const float* quats, // [P, 4]
    const float* opacities, // [P]
    const float* shs, // [P, MAX_COEFFS, NUM_CHANNELS]
    // trans mats 
    const float* viewmatrix, // [16]
    const float* projmatrix, // [16]
    const float* cam_pos, // [3]
    // output 
    float* depths, // [P]
    int* radii, // [P]
    int* p_images, // [P, 2]
    float* conic_opacities, // [P, 4] 
    int* touched_tiles // [P]
)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= P) return;

    radii[idx] = 0; 
    touched_tiles[idx] = 0; 

    // project and culling 
    const float* p_world = & orig_points[3 * idx]; 
    float p_view[3]; 
    float p_ndc[3]; 
    int p_image[2]; 

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
    float radius_f = compute_gaussian_radius_from_cov2D(cov2D); 
    int radius_px = (int)ceilf(radius_f);

    int tile_min[2]; 
    int tile_max[2]; 
    // if the coverage area is zero, skip this gauss 
    if (!compute_tile_rect_from_pixel_radius(
        p_image, radius_px, tile_grid, tile_min, tile_max 
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

    // save results 
    depths[idx] = p_view[2]; 
    radii[idx] = radius_px; 
    p_images[idx * 2 + 0] = p_image[0]; 
    p_images[idx * 2 + 1] = p_image[1]; 
    float opacity = opacities[idx]; 
    conic_opacities[idx * 4 + 0] = conic[0]; 
    conic_opacities[idx * 4 + 1] = conic[1]; 
    conic_opacities[idx * 4 + 2] = conic[2]; 
    conic_opacities[idx * 4 + 3] = opacity * opacity_scale; 
    touched_tiles[idx] = (tile_max[1] - tile_min[1]) * (tile_max[0] - tile_min[0]); 
}