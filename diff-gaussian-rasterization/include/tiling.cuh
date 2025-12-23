// include/tiling.cuh 
#pragma once 
#include "config.h" 
#include <cstdint>


__device__ __forceinline__
bool compute_tile_rect_from_pixel_radius(
    const float* p_image,   // [x, y] in pixel space
    const float radius, 
    const dim3 tile_grid, 
    uint2* tile_min, 
    uint2* tile_max 
)
{
    float px = p_image[0], py = p_image[1]; 

    float xmin = px - radius; 
    float xmax = px + radius; 
    float ymin = py - radius; 
    float ymax = py + radius; 

    // convert to tile indices
    int tx_min = (int)floorf(xmin / BLOCK_X);
    int ty_min = (int)floorf(ymin / BLOCK_Y);
    int tx_max = (int)floorf((xmax + BLOCK_X - 1) / BLOCK_X);
    int ty_max = (int)floorf((ymax + BLOCK_Y - 1) / BLOCK_Y);

    // clamp to grid
    tx_min = max(0, min(tx_min, (int)tile_grid.x));
    ty_min = max(0, min(ty_min, (int)tile_grid.y));
    tx_max = max(0, min(tx_max, (int)tile_grid.x));
    ty_max = max(0, min(ty_max, (int)tile_grid.y));

    // empty coverage check
    if (tx_min >= tx_max || ty_min >= ty_max)
        return false;

    tile_min->x = (uint32_t)tx_min;
    tile_min->y = (uint32_t)ty_min;
    tile_max->x = (uint32_t)tx_max;
    tile_max->y = (uint32_t)ty_max;

    return true; 
}