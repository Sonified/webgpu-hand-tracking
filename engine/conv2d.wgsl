// Fused Conv2D + PReLU compute shader.
// Handles both regular and depthwise convolution via the `group` uniform.
// When group == channels_in, it's depthwise. When group == 1, it's standard.
//
// This is the core building block of BlazePalm. The entire backbone is
// repeated applications of: DepthwiseConv5x5 -> Conv1x1 -> Add -> PReLU.
// Fusing Conv + PReLU into one shader eliminates an intermediate buffer
// and a GPU dispatch. The Add (residual connection) is handled by the
// caller passing the residual buffer as an additional binding.

// Uniforms: shape parameters for this particular conv layer.
struct ConvParams {
    batch: u32,
    in_c: u32,        // input channels
    in_h: u32,        // input height
    in_w: u32,        // input width
    out_c: u32,       // output channels
    out_h: u32,       // output height
    out_w: u32,       // output width
    kern_h: u32,      // kernel height
    kern_w: u32,      // kernel width
    stride_h: u32,
    stride_w: u32,
    pad_top: u32,
    pad_left: u32,
    group: u32,       // 1 = standard conv, in_c = depthwise
    has_prelu: u32,   // 0 = no activation, 1 = PReLU
    has_residual: u32, // 0 = no residual add, 1 = add residual before activation
}

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;  // conv kernel weights
@group(0) @binding(3) var<storage, read> bias: array<f32>;     // conv bias
@group(0) @binding(4) var<storage, read> prelu_slope: array<f32>; // per-channel PReLU slopes
@group(0) @binding(5) var<storage, read> residual: array<f32>;  // residual connection input
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ow = gid.x;  // output width position
    let oh = gid.y;  // output height position
    let oc = gid.z;  // output channel

    // Bounds check
    if (ow >= params.out_w || oh >= params.out_h || oc >= params.out_c) {
        return;
    }

    // Compute the convolution
    let channels_per_group = params.in_c / params.group;
    let group_id = oc / (params.out_c / params.group);
    let in_c_start = group_id * channels_per_group;

    var sum: f32 = bias[oc];

    for (var ic: u32 = 0u; ic < channels_per_group; ic++) {
        let in_c_idx = in_c_start + ic;
        for (var kh: u32 = 0u; kh < params.kern_h; kh++) {
            for (var kw: u32 = 0u; kw < params.kern_w; kw++) {
                let ih = oh * params.stride_h + kh - params.pad_top;
                let iw = ow * params.stride_w + kw - params.pad_left;

                // Boundary check (unsigned underflow handles negative coords)
                if (ih < params.in_h && iw < params.in_w) {
                    let in_idx = in_c_idx * params.in_h * params.in_w + ih * params.in_w + iw;
                    // Weight layout: [out_c, in_c_per_group, kern_h, kern_w]
                    let w_idx = oc * channels_per_group * params.kern_h * params.kern_w
                              + ic * params.kern_h * params.kern_w
                              + kh * params.kern_w + kw;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }
    }

    // Residual add (before activation, matching the graph structure)
    if (params.has_residual == 1u) {
        let out_idx = oc * params.out_h * params.out_w + oh * params.out_w + ow;
        sum += residual[out_idx];
    }

    // PReLU activation: x > 0 ? x : slope * x
    // One line. No decomposition. No 4-op workaround.
    if (params.has_prelu == 1u) {
        if (sum < 0.0) {
            sum = sum * prelu_slope[oc];
        }
    }

    // Write output (NCHW layout)
    let out_idx = oc * params.out_h * params.out_w + oh * params.out_w + ow;
    output[out_idx] = sum;
}
