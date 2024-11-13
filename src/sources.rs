/// Name of the kernel doing the multiplication (function name)
pub const KERNEL_NAME: &str = "mul";

/// tile size for various implementations
pub const TILE: usize = 16;
/// how much elements a thread is counting in [HARD] mode
pub const ELEM_PER_THREAD: usize = 2;

/// Source opencl code for easy multiplication
pub const EASY_SOURCE: &str = r#"
void kernel mul(global const float* m1, global const float* m2, 
                              global float* m3, const uint n, const uint m, const uint k) {
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    float sum = 0.0f;
    for (uint w = 0; w < k; w++) {
        sum += m1[j * k + w] * m2[w * n + i];
    }
    m3[j * n + i] = sum;
}"#;

/// Source opencl code for medium multiplication
pub const MEDIUM_MUL: &str = r#"
#define TILE 16

kernel void mul(global float* m1, global float* m2, global float* m3, uint n, uint m, uint k) {

    uint i = get_global_id(0);
    uint j = get_global_id(1);

    uint li = get_local_id(0);
    uint lj = get_local_id(1);

    local float la[TILE][TILE];
    local float lb[TILE][TILE];

    float sum = 0.0f;
    uint iter = k / TILE;
    for (uint w = 0; w < iter; w++) {
        uint trow = TILE * w + li;
        uint tcol = TILE * w + lj;
        la[lj][li] = m1[j * k + trow];
        lb[lj][li] = m2[tcol * n + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int z = 0; z < TILE; z++) {
            sum += la[lj][z] * lb[z][li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    m3[j * n + i] = sum;
}"#;

/// Source opencl code for hard multiplication
pub const HARD_MUL: &str = r#"
#define TILE 16
#define ELEM_PER_THREAD 2

// TILE / ELEM_PER_THREAD
#define NEW_TILE_SIZE 8

kernel void mul(const global float* m1, const global float* m2, global float* m3, int n, int m, int k) {
    uint li = get_local_id(0);
    uint lj = get_local_id(1);

    uint i = TILE * get_group_id(0) + li;
    uint j = TILE * get_group_id(1) + lj;

    local float la[TILE][TILE];
    local float lb[TILE][TILE];
    
    float2 acc = (float2)(0.0f);

    uint iter = k / TILE;
    for (uint t = 0; t < iter; t++) {
        uint trow = TILE * t + li;
        uint tcol = TILE * t + lj;

        for (uint w = 0; w < ELEM_PER_THREAD; w++) {
            la[lj + w * NEW_TILE_SIZE][li] = m2[(tcol + w * NEW_TILE_SIZE) * n + i];
            lb[lj + w * NEW_TILE_SIZE][li] = m1[(j + w * NEW_TILE_SIZE) * k + trow];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (uint kk = 0; kk < TILE; kk++) {
            acc += (float2)(la[kk][li]) * (float2)(lb[lj][kk], lb[lj + NEW_TILE_SIZE][kk]);
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    m3[j * n + i] = acc.s0;
    m3[(j + NEW_TILE_SIZE) * n + i] = acc.s1;
}
"#;
