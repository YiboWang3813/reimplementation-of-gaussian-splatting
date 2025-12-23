// include/tiling.cuh 
#pragma once 
#include "config.h" 
#include <cstdint>


__device__ __forceinline__
bool compute_tile_rect_from_pixel_radius(
    const int* p_image,   // [2] pixel center
    int radius_px,          // integer pixel radius
    dim3 tile_grid,
    int* tile_min,          // [2]
    int* tile_max           // [2]
)
{
    int px = p_image[0];
    int py = p_image[1];

    // integer pixel bounds (conceptually)
    float xmin = px - radius_px;
    float xmax = px + radius_px;
    float ymin = py - radius_px;
    float ymax = py + radius_px;

    // convert to tile indices
    int tx_min = xmin / BLOCK_X;
    int ty_min = ymin / BLOCK_Y;
    int tx_max = (xmax + BLOCK_X) / BLOCK_X;
    int ty_max = (ymax + BLOCK_Y) / BLOCK_Y;
    
    // clamp to grid
    tx_min = max(0, min(tx_min, (int)tile_grid.x));
    ty_min = max(0, min(ty_min, (int)tile_grid.y));
    tx_max = max(0, min(tx_max, (int)tile_grid.x));
    ty_max = max(0, min(ty_max, (int)tile_grid.y));

    if (tx_min >= tx_max || ty_min >= ty_max)
        return false;

    tile_min[0] = tx_min;
    tile_min[1] = ty_min;
    tile_max[0] = tx_max;
    tile_max[1] = ty_max;

    return true;
}
