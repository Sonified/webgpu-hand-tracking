// Bilinear 2x upsample compute shader.
// Used in the Feature Pyramid Network (FPN) head to upsample feature maps
// before merging with the previous stage's output.

struct ResizeParams {
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,  // in_h * 2
    out_w: u32,  // in_w * 2
}

@group(0) @binding(0) var<uniform> params: ResizeParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ox = gid.x;
    let oy = gid.y;
    let c = gid.z;

    if (ox >= params.out_w || oy >= params.out_h || c >= params.channels) {
        return;
    }

    // Map output coordinate to input space
    let scale_h = f32(params.in_h) / f32(params.out_h);
    let scale_w = f32(params.in_w) / f32(params.out_w);
    // half_pixel coordinate transform: clamp to [0, in_size-1] BEFORE
    // computing the integer/fractional split so edge pixels don't get
    // blended with out-of-bounds coordinates.
    let iy_raw = (f32(oy) + 0.5) * scale_h - 0.5;
    let ix_raw = (f32(ox) + 0.5) * scale_w - 0.5;
    let iy_f = max(iy_raw, 0.0);
    let ix_f = max(ix_raw, 0.0);

    let iy0 = u32(floor(iy_f));
    let ix0 = u32(floor(ix_f));
    let iy1 = min(iy0 + 1u, params.in_h - 1u);
    let ix1 = min(ix0 + 1u, params.in_w - 1u);

    let fy = iy_f - floor(iy_f);
    let fx = ix_f - floor(ix_f);

    let base = c * params.in_h * params.in_w;
    let v00 = input[base + iy0 * params.in_w + ix0];
    let v01 = input[base + iy0 * params.in_w + ix1];
    let v10 = input[base + iy1 * params.in_w + ix0];
    let v11 = input[base + iy1 * params.in_w + ix1];

    let val = v00 * (1.0 - fy) * (1.0 - fx)
            + v01 * (1.0 - fy) * fx
            + v10 * fy * (1.0 - fx)
            + v11 * fy * fx;

    output[c * params.out_h * params.out_w + oy * params.out_w + ox] = val;
}
