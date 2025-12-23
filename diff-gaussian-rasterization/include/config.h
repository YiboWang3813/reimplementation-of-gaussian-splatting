#pragma once

#include <cuda_runtime.h>

#define prefiltered true 
#define near_z 0.2f 

#define antialiasing true 

#define NUM_CHANNELS 3 // Default 3, RGB


#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_Z 1

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

#define BLOCK_2D dim3(BLOCK_X, BLOCK_Y, BLOCK_Z)

#define GRID_2D(width, height) \
    dim3(DIV_UP((width), BLOCK_X), \
         DIV_UP((height), BLOCK_Y), \
         1)