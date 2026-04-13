// MaxPool 2x2 stride 2 compute shader.
// Used at stage boundaries to downsample spatial dimensions by 2x.
// Fused with the channel padding that follows in the graph (Pad op
// pads channels with zeros to match the next stage's wider channel count).

struct PoolParams {
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,     // in_h / 2
    out_w: u32,     // in_w / 2
    out_channels: u32, // channels after padding (may be > channels)
}

@group(0) @binding(0) var<uniform> params: PoolParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ow = gid.x;
    let oh = gid.y;
    let oc = gid.z;

    if (ow >= params.out_w || oh >= params.out_h || oc >= params.out_channels) {
        return;
    }

    let out_idx = oc * params.out_h * params.out_w + oh * params.out_w + ow;

    // Channels beyond the input channel count are zero-padding
    if (oc >= params.channels) {
        output[out_idx] = 0.0;
        return;
    }

    // 2x2 max pool
    let ih = oh * 2u;
    let iw = ow * 2u;
    let base = oc * params.in_h * params.in_w;

    var maxval: f32 = input[base + ih * params.in_w + iw];
    maxval = max(maxval, input[base + ih * params.in_w + iw + 1u]);
    maxval = max(maxval, input[base + (ih + 1u) * params.in_w + iw]);
    maxval = max(maxval, input[base + (ih + 1u) * params.in_w + iw + 1u]);

    output[out_idx] = maxval;
}
