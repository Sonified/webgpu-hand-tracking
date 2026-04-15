// Fused Conv2D + PReLU compute shader.
// Handles both regular and depthwise convolution via the `group` uniform.
// When group == channels_in, it's depthwise. When group == 1, it's standard.
//
// Optimizations:
// - Workgroup shared memory tile for depthwise conv (each input pixel loaded once)
// - vec4 accumulation for 1x1 pointwise conv

struct ConvParams {
    batch: u32,
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    kern_h: u32,
    kern_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_top: u32,
    pad_left: u32,
    group: u32,       // 1 = standard conv, in_c = depthwise
    has_prelu: u32,   // 0 = none, 1 = PReLU, 2 = ReLU6, 3 = ReLU
    has_residual: u32,
}

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> prelu_slope: array<f32>;
@group(0) @binding(5) var<storage, read> residual: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

// Shared memory tile for depthwise conv.
// Worst case: 8x8 output with stride 2 and 5x5 kernel = (8*2+4)x(8*2+4) = 20x20 = 400
var<workgroup> tile: array<f32, 400>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u
) {
    let ow = gid.x;
    let oh = gid.y;
    let oc = gid.z;

    let channels_per_group = params.in_c / params.group;
    let group_id = oc / (params.out_c / params.group);
    let in_c_start = group_id * channels_per_group;
    let in_bounds = ow < params.out_w && oh < params.out_h && oc < params.out_c;

    var sum: f32 = 0.0;
    if (in_bounds) { sum = bias[oc]; }

    // Depthwise path: use shared memory tile
    // ALL threads in workgroup must participate in barrier even if out of bounds
    if (params.group == params.in_c && params.kern_h <= 5u && params.kern_w <= 5u) {
        // Tile covers the 8x8 output region scaled by stride, plus kernel halo
        let tile_w: u32 = 8u * params.stride_w + params.kern_w - 1u;
        let tile_h: u32 = 8u * params.stride_h + params.kern_h - 1u;
        // Use signed arithmetic to avoid unsigned underflow when pad > origin
        let origin_h: i32 = i32(wgid.y * 8u * params.stride_h) - i32(params.pad_top);
        let origin_w: i32 = i32(wgid.x * 8u * params.stride_w) - i32(params.pad_left);
        let local_idx = lid.y * 8u + lid.x;
        let tile_size = tile_w * tile_h;
        let in_c_idx = oc; // depthwise: output channel == input channel

        for (var t = local_idx; t < tile_size; t += 64u) {
            let ty = i32(t / tile_w);
            let tx = i32(t % tile_w);
            let iy = origin_h + ty;
            let ix = origin_w + tx;
            var val: f32 = 0.0;
            if (iy >= 0 && iy < i32(params.in_h) && ix >= 0 && ix < i32(params.in_w)) {
                val = input[in_c_idx * params.in_h * params.in_w + u32(iy) * params.in_w + u32(ix)];
            }
            tile[t] = val;
        }
        workgroupBarrier();

        if (in_bounds) {
            let local_oh = lid.y * params.stride_h;
            let local_ow = lid.x * params.stride_w;
            for (var kh: u32 = 0u; kh < params.kern_h; kh++) {
                for (var kw: u32 = 0u; kw < params.kern_w; kw++) {
                    let tile_idx = (local_oh + kh) * tile_w + (local_ow + kw);
                    let w_idx = oc * params.kern_h * params.kern_w + kh * params.kern_w + kw;
                    sum += tile[tile_idx] * weights[w_idx];
                }
            }
        }
    }
    else if (!in_bounds) {
        // Out of bounds -- skip all compute paths
    }
    // 1x1 pointwise path: vec4 accumulation (no shared memory -- M1 L2 cache handles weight reuse)
    else if (params.kern_h == 1u && params.kern_w == 1u) {
        let spatial_idx = oh * params.in_w + ow;
        let cpg = channels_per_group;
        let cpg4 = cpg / 4u;
        let w_base = oc * cpg;

        for (var ic4: u32 = 0u; ic4 < cpg4; ic4++) {
            let ic = in_c_start + ic4 * 4u;
            let i0 = ic * params.in_h * params.in_w + spatial_idx;
            let stride = params.in_h * params.in_w;
            let v = vec4f(input[i0], input[i0 + stride], input[i0 + stride * 2u], input[i0 + stride * 3u]);
            let w = vec4f(weights[w_base + ic4 * 4u], weights[w_base + ic4 * 4u + 1u], weights[w_base + ic4 * 4u + 2u], weights[w_base + ic4 * 4u + 3u]);
            sum += dot(v, w);
        }
        for (var ic = in_c_start + cpg4 * 4u; ic < in_c_start + cpg; ic++) {
            let in_idx = ic * params.in_h * params.in_w + spatial_idx;
            let w_idx = oc * cpg + (ic - in_c_start);
            sum += input[in_idx] * weights[w_idx];
        }
    }
    // General path: standard nested loops
    else {
        for (var ic: u32 = 0u; ic < channels_per_group; ic++) {
            let in_c_idx = in_c_start + ic;
            for (var kh: u32 = 0u; kh < params.kern_h; kh++) {
                for (var kw: u32 = 0u; kw < params.kern_w; kw++) {
                    let ih = oh * params.stride_h + kh - params.pad_top;
                    let iw = ow * params.stride_w + kw - params.pad_left;
                    if (ih < params.in_h && iw < params.in_w) {
                        let in_idx = in_c_idx * params.in_h * params.in_w + ih * params.in_w + iw;
                        let w_idx = oc * channels_per_group * params.kern_h * params.kern_w
                                  + ic * params.kern_h * params.kern_w
                                  + kh * params.kern_w + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
    }

    if (in_bounds) {
        // Residual add
        if (params.has_residual == 1u) {
            let out_idx = oc * params.out_h * params.out_w + oh * params.out_w + ow;
            sum += residual[out_idx];
        }

        // Activation
        if (params.has_prelu == 1u) {
            if (sum < 0.0) { sum = sum * prelu_slope[oc]; }
        } else if (params.has_prelu == 2u) {
            sum = clamp(sum, 0.0, 6.0);
        } else if (params.has_prelu == 3u) {
            sum = max(sum, 0.0);
        }

        let out_idx = oc * params.out_h * params.out_w + oh * params.out_w + ow;
        output[out_idx] = sum;
    }
}
