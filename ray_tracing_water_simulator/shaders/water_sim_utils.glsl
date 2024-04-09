vec4 pack_float32(const in float val)
{
    const vec4 bit_shift = vec4(256.0*256.0*256.0, 256.0*256.0, 256.0, 1.0);
    const vec4 bit_mask  = vec4(0.0, 1.0/256.0, 1.0/256.0, 1.0/256.0);
    vec4 res = fract(val * bit_shift);
    res -= res.xxyz * bit_mask;
    return res;
}

float unpack_float32(const in vec4 rgba)
{
    const vec4 bit_shift = vec4(1.0/(256.0*256.0*256.0), 1.0/(256.0*256.0), 1.0/256.0, 1.0);
    return float(dot(rgba, bit_shift));
}

float compute_weight(float x)
{
    if(x > 0.5 && x<1.5)
    {
        float t = 1.5 - x;
        return 0.5 * t * t;
    }
    if(x > 0 && x<0.5)
    {
        return fma(-x,x,0.75);
    }
    return 0;
}

vec3 get_grid_coord(vec3 pos,int grid_size)
{
    vec3 gridCoord = pos - vec3(-4.9); // [0, 10]
    gridCoord /= 10.0; //[0, 1]
    gridCoord *= (grid_size - 1);  // [0, grid_size]
    gridCoord += vec3(0.5);
    return gridCoord;
}

float packIntsToFloat(int int1, int int2) {
    return float(int1 << 8 | int2);
}

void unpackFloatToInts(float packedFloat, out int int1, out int int2) {
    int1 = int(packedFloat) >> 8;
    int2 = int(packedFloat) & 0xFF;
}


