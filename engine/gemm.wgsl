// General Matrix Multiply (fully-connected layer).
// Computes: output = input * weights + bias
// Used at the end of the hand landmark model for the classification/regression heads.

struct GemmParams {
    M: u32,  // rows of input (batch, typically 1)
    K: u32,  // columns of input = rows of weights
    N: u32,  // columns of weights = output features
    has_bias: u32,
    has_sigmoid: u32, // 0 = none, 1 = sigmoid activation
}

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;    // [M, K]
@group(0) @binding(2) var<storage, read> weights: array<f32>;  // [K, N] (not transposed)
@group(0) @binding(3) var<storage, read> bias: array<f32>;     // [N]
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [M, N]

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let col = gid.x;  // output column (feature)
    let row = gid.y;  // output row (batch)

    if (col >= params.N || row >= params.M) {
        return;
    }

    var sum: f32 = 0.0;
    if (params.has_bias == 1u) {
        sum = bias[col];
    }

    for (var k: u32 = 0u; k < params.K; k++) {
        sum += input[row * params.K + k] * weights[k * params.N + col];
    }

    if (params.has_sigmoid == 1u) {
        sum = 1.0 / (1.0 + exp(-sum));
    }

    output[row * params.N + col] = sum;
}
