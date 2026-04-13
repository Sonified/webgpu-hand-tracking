// Global Average Pooling: average all spatial positions per channel.
// Input: [1, C, H, W] -> Output: [1, C, 1, 1] (flattened to [C])
// Used once in the hand landmark model before the fully-connected heads.

struct PoolParams {
    channels: u32,
    height: u32,
    width: u32,
}

@group(0) @binding(0) var<uniform> params: PoolParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let c = gid.x;
    if (c >= params.channels) { return; }

    let spatial = params.height * params.width;
    let base = c * spatial;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < spatial; i++) {
        sum += input[base + i];
    }
    output[c] = sum / f32(spatial);
}
