// Fused residual block: DW Conv -> 1x1 Conv -> [Residual Add] -> Activation
// Single dispatch replaces 3-4 separate dispatches.
// DW output never materializes to memory -- computed in registers and immediately
// accumulated into the 1x1 conv output.

struct BlockDesc {
    dw_in_ch: u32,
    dw_kern: u32,          // kernel size (3 or 5)
    dw_stride: u32,
    dw_pad_t: u32,         // asymmetric padding: top
    dw_pad_l: u32,         // left
    dw_pad_b: u32,         // bottom
    dw_pad_r: u32,         // right
    dw_w_off: u32,         // offset into weights[] for DW kernel
    dw_b_off: u32,         // offset into weights[] for DW bias
    pw_out_ch: u32,        // 1x1 output channels
    pw_w_off: u32,         // offset into weights[] for 1x1 kernel
    pw_b_off: u32,         // offset into weights[] for 1x1 bias
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    has_residual: u32,     // 0=none, 1=same-ch add, 2=padded-ch add
    res_ch: u32,           // residual input channels (before padding)
    act_type: u32,         // 0=none, 1=PReLU, 2=ReLU6, 3=ReLU
    act_off: u32,          // offset into weights[] for PReLU slopes
}

@group(0) @binding(0) var<uniform> d: BlockDesc;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ow = gid.x;
    let oh = gid.y;
    let oc = gid.z;

    if (ow >= d.out_w || oh >= d.out_h || oc >= d.pw_out_ch) { return; }

    // Start with 1x1 bias
    var pw_sum: f32 = weights[d.pw_b_off + oc];

    // Fused DW + 1x1: for each input channel, compute DW at (oh,ow),
    // then immediately multiply by 1x1 weight and accumulate.
    let kern = d.dw_kern;
    for (var ic: u32 = 0u; ic < d.dw_in_ch; ic++) {
        var dw_val: f32 = weights[d.dw_b_off + ic];
        for (var kh: u32 = 0u; kh < kern; kh++) {
            for (var kw: u32 = 0u; kw < kern; kw++) {
                // Apply asymmetric padding
                let ih_padded = oh * d.dw_stride + kh;
                let iw_padded = ow * d.dw_stride + kw;
                let ih = ih_padded - d.dw_pad_t;
                let iw = iw_padded - d.dw_pad_l;
                // Unsigned comparison: if ih wrapped negative it becomes huge (> in_h)
                if (ih < d.in_h && iw < d.in_w) {
                    let in_idx = ic * d.in_h * d.in_w + ih * d.in_w + iw;
                    let w_idx = d.dw_w_off + ic * kern * kern + kh * kern + kw;
                    dw_val += input[in_idx] * weights[w_idx];
                }
            }
        }

        // Immediately accumulate into 1x1 output
        let pw_w_idx = d.pw_w_off + oc * d.dw_in_ch + ic;
        pw_sum += dw_val * weights[pw_w_idx];
    }

    // Residual connection
    if (d.has_residual >= 1u) {
        let sp = oh * d.out_w + ow;
        if (d.has_residual == 2u) {
            if (oc < d.res_ch) {
                pw_sum += residual[oc * d.out_h * d.out_w + sp];
            }
        } else {
            pw_sum += residual[oc * d.out_h * d.out_w + sp];
        }
    }

    // Activation
    if (d.act_type == 1u) {
        if (pw_sum < 0.0) { pw_sum *= weights[d.act_off + oc]; }
    } else if (d.act_type == 2u) {
        pw_sum = clamp(pw_sum, 0.0, 6.0);
    } else if (d.act_type == 3u) {
        pw_sum = max(pw_sum, 0.0);
    }

    output[oc * d.out_h * d.out_w + oh * d.out_w + ow] = pw_sum;
}
