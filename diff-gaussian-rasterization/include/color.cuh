#pragma once 


__device__ __constant__ float SH_C0 = 0.28209479177387814f;
__device__ __constant__ float SH_C1 = 0.4886025119029199f;

__device__ __constant__ float SH_C2[5] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};

__device__ __constant__ float SH_C3[7] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};


__device__ __forceinline__
void compute_color_from_sh(
    int sh_deg,
    int max_coeffs,
    const float* p_world,    // [3]
    const float* cam_pos,    // [3]
    const float* sh,         // [max_coeffs, 3]
    float* rgb_out,          // [3]
    bool* clamped            // [3]
)
{
    // view direction = normalize(p_world - cam_pos)
    float dx = p_world[0] - cam_pos[0];
    float dy = p_world[1] - cam_pos[1];
    float dz = p_world[2] - cam_pos[2];

    float inv_len = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-8f);
    dx *= inv_len;
    dy *= inv_len;
    dz *= inv_len;

    // degree 0
    float r = SH_C0 * sh[0 * 3 + 0];
    float g = SH_C0 * sh[0 * 3 + 1];
    float b = SH_C0 * sh[0 * 3 + 2];

    if (sh_deg > 0)
    {
        r += -SH_C1 * dy * sh[1 * 3 + 0]
           +  SH_C1 * dz * sh[2 * 3 + 0]
           -  SH_C1 * dx * sh[3 * 3 + 0];

        g += -SH_C1 * dy * sh[1 * 3 + 1]
           +  SH_C1 * dz * sh[2 * 3 + 1]
           -  SH_C1 * dx * sh[3 * 3 + 1];

        b += -SH_C1 * dy * sh[1 * 3 + 2]
           +  SH_C1 * dz * sh[2 * 3 + 2]
           -  SH_C1 * dx * sh[3 * 3 + 2];

        if (sh_deg > 1)
        {
            float xx = dx * dx, yy = dy * dy, zz = dz * dz;
            float xy = dx * dy, yz = dy * dz, xz = dx * dz;

            r += SH_C2[0] * xy * sh[4 * 3 + 0]
               + SH_C2[1] * yz * sh[5 * 3 + 0]
               + SH_C2[2] * (2.f * zz - xx - yy) * sh[6 * 3 + 0]
               + SH_C2[3] * xz * sh[7 * 3 + 0]
               + SH_C2[4] * (xx - yy) * sh[8 * 3 + 0];

            g += SH_C2[0] * xy * sh[4 * 3 + 1]
               + SH_C2[1] * yz * sh[5 * 3 + 1]
               + SH_C2[2] * (2.f * zz - xx - yy) * sh[6 * 3 + 1]
               + SH_C2[3] * xz * sh[7 * 3 + 1]
               + SH_C2[4] * (xx - yy) * sh[8 * 3 + 1];

            b += SH_C2[0] * xy * sh[4 * 3 + 2]
               + SH_C2[1] * yz * sh[5 * 3 + 2]
               + SH_C2[2] * (2.f * zz - xx - yy) * sh[6 * 3 + 2]
               + SH_C2[3] * xz * sh[7 * 3 + 2]
               + SH_C2[4] * (xx - yy) * sh[8 * 3 + 2];

            if (sh_deg > 2)
            {
                r += SH_C3[0] * dy * (3.f * xx - yy) * sh[9  * 3 + 0]
                   + SH_C3[1] * xy * dz              * sh[10 * 3 + 0]
                   + SH_C3[2] * dy * (4.f * zz - xx - yy) * sh[11 * 3 + 0]
                   + SH_C3[3] * dz * (2.f * zz - 3.f * xx - 3.f * yy) * sh[12 * 3 + 0]
                   + SH_C3[4] * dx * (4.f * zz - xx - yy) * sh[13 * 3 + 0]
                   + SH_C3[5] * dz * (xx - yy) * sh[14 * 3 + 0]
                   + SH_C3[6] * dx * (xx - 3.f * yy) * sh[15 * 3 + 0];

                g += SH_C3[0] * dy * (3.f * xx - yy) * sh[9  * 3 + 1]
                   + SH_C3[1] * xy * dz              * sh[10 * 3 + 1]
                   + SH_C3[2] * dy * (4.f * zz - xx - yy) * sh[11 * 3 + 1]
                   + SH_C3[3] * dz * (2.f * zz - 3.f * xx - 3.f * yy) * sh[12 * 3 + 1]
                   + SH_C3[4] * dx * (4.f * zz - xx - yy) * sh[13 * 3 + 1]
                   + SH_C3[5] * dz * (xx - yy) * sh[14 * 3 + 1]
                   + SH_C3[6] * dx * (xx - 3.f * yy) * sh[15 * 3 + 1];

                b += SH_C3[0] * dy * (3.f * xx - yy) * sh[9  * 3 + 2]
                   + SH_C3[1] * xy * dz              * sh[10 * 3 + 2]
                   + SH_C3[2] * dy * (4.f * zz - xx - yy) * sh[11 * 3 + 2]
                   + SH_C3[3] * dz * (2.f * zz - 3.f * xx - 3.f * yy) * sh[12 * 3 + 2]
                   + SH_C3[4] * dx * (4.f * zz - xx - yy) * sh[13 * 3 + 2]
                   + SH_C3[5] * dz * (xx - yy) * sh[14 * 3 + 2]
                   + SH_C3[6] * dx * (xx - 3.f * yy) * sh[15 * 3 + 2];
            }
        }
    }

    // bias
    r += 0.5f;
    g += 0.5f;
    b += 0.5f;

    // clamp + record
    clamped[0] = (r < 0.0f);
    clamped[1] = (g < 0.0f);
    clamped[2] = (b < 0.0f);

    rgb_out[0] = fmaxf(r, 0.0f);
    rgb_out[1] = fmaxf(g, 0.0f);
    rgb_out[2] = fmaxf(b, 0.0f);
}
