// Element-wise addition: output[i] = a[i] + b[i]
// Used for residual connections in MobileNetV2 inverted residual blocks.

struct AddParams {
    count: u32,  // total number of floats
}

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.count) { return; }
    output[i] = a[i] + b[i];
}
