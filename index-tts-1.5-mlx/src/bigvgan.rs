//! BigVGAN vocoder with ECAPA-TDNN speaker encoder for IndexTTS 1.5.
//!
//! Architecture overview:
//!   1. Reference mel → ECAPA-TDNN → speaker embedding [B, spk_dim]
//!   2. GPT latent [B, T, gpt_dim] → conv_pre → [B, T, ch_init]
//!   3. Global conditioning: cond_layer(spk) → FiLM scale+shift
//!   4. N upsampling stages (ConvTranspose1d + per-stage FiLM + MRF)
//!   5. SnakeBeta → conv_post → tanh → waveform [B, T', 1]
//!
//! All intermediate tensors use MLX channels-last layout: [B, L, C].
//!
//! Key prefixes in model.safetensors:
//!   bigvgan.conv_pre.*           — initial Conv1d (gpt_dim → ch_init)
//!   bigvgan.speaker_encoder.*    — ECAPA-TDNN
//!   bigvgan.cond_layer.*         — Linear(spk_dim → 2*ch_init)
//!   bigvgan.conds.{i}.*          — Linear(spk_dim → 2*ch_i) per stage
//!   bigvgan.ups.{i}.*            — ConvTranspose1d
//!   bigvgan.resblocks.{i}.*      — MRF residual blocks (SnakeBeta activations)
//!   bigvgan.conv_post.*          — final Conv1d (last_ch → 1)
//!   bigvgan.activation_post.*    — SnakeBeta params for post-conv
//!
//! NOTE: ECAPA-TDNN key names follow the speechbrain/WeSpeaker convention.
//!       Verify against actual safetensors keys on first Mac run.

use std::collections::HashMap;

use mlx_rs::{array, nn, ops, Array};

use crate::conditioner::{get_weight, load_linear};
use crate::config::BigvganConfig;
use crate::error::{Error, Result};

// ─── SnakeBeta activation ─────────────────────────────────────────────────────

/// SnakeBeta: y = x + (1/β)·sin²(β·x)
///
/// `alpha` and `beta` are [C] per-channel learned params.
/// Input x is [B, L, C] (channels-last).
fn snake_beta(x: &Array, alpha: &Array, beta: &Array) -> Result<Array> {
    let c = alpha.shape()[0];
    // Reshape [C] → [1, 1, C] to broadcast over [B, L, C]
    let b = beta.reshape(&[1, 1, c])?;
    let bx = (&b * x)?;
    let sin_bx = ops::sin(&bx)?;
    let sin2_bx = (&sin_bx * &sin_bx)?;
    // sin²(β·x) / β
    let term = (&sin2_bx / &b)?;
    (x + &term)
}

// ─── BatchNorm1d (inference only) ────────────────────────────────────────────

/// BatchNorm1d in inference mode (uses running statistics).
/// Input/output shape: [B, L, C] (channels-last).
struct BatchNorm1d {
    weight: Array,       // gamma [C]
    bias: Array,         // beta  [C]
    running_mean: Array, // [C]
    running_var: Array,  // [C]
    eps: f32,
}

impl BatchNorm1d {
    fn load(weights: &HashMap<String, Array>, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, &format!("{prefix}.weight"))?,
            bias: get_weight(weights, &format!("{prefix}.bias"))?,
            running_mean: get_weight(weights, &format!("{prefix}.running_mean"))?,
            running_var: get_weight(weights, &format!("{prefix}.running_var"))?,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let c = self.weight.shape()[0];
        let mean = self.running_mean.reshape(&[1, 1, c])?;
        let var = self.running_var.reshape(&[1, 1, c])?;
        let w = self.weight.reshape(&[1, 1, c])?;
        let b = self.bias.reshape(&[1, 1, c])?;
        let std = ops::sqrt(&(&var + self.eps)?)?;
        let normalized = ((x - &mean)? / &std)?;
        ((&normalized * &w)? + &b)
    }
}

// ─── ECAPA-TDNN ──────────────────────────────────────────────────────────────

/// Squeeze-and-Excitation module inside SE-Res2Net blocks.
struct SeModule {
    fc1_w: Array,
    fc1_b: Option<Array>,
    fc2_w: Array,
    fc2_b: Option<Array>,
}

impl SeModule {
    fn load(weights: &HashMap<String, Array>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1_w: get_weight(weights, &format!("{prefix}.fc1.weight"))?,
            fc1_b: weights.get(&format!("{prefix}.fc1.bias")).cloned(),
            fc2_w: get_weight(weights, &format!("{prefix}.fc2.weight"))?,
            fc2_b: weights.get(&format!("{prefix}.fc2.bias")).cloned(),
        })
    }

    /// x: [B, L, C] → [B, L, C]  (element-wise SE scaling)
    fn forward(&self, x: &Array) -> Result<Array> {
        // Global average pool: [B, L, C] → [B, 1, C]
        let pooled = x.mean_axis(1, true)?;

        // Linear + ReLU + Linear + Sigmoid
        let h = apply_linear_bias(ops::matmul(&pooled, &self.fc1_w.t())?, &self.fc1_b)?;
        let h = ops::relu(&h)?;
        let h = apply_linear_bias(ops::matmul(&h, &self.fc2_w.t())?, &self.fc2_b)?;
        let scale = ops::sigmoid(&h)?;
        (x * &scale)
    }
}

/// SE-Res2Net block (Bottle2neck) used in ECAPA-TDNN.
struct Bottle2neck {
    conv1_w: Array,
    conv1_b: Option<Array>,
    bn1: BatchNorm1d,
    branch_convs: Vec<Array>,
    branch_biases: Vec<Option<Array>>,
    bn2: BatchNorm1d,
    conv3_w: Array,
    conv3_b: Option<Array>,
    bn3: BatchNorm1d,
    se: SeModule,
    scale: usize,
    width: i32,
    dilation: usize,
    kernel: usize,
}

impl Bottle2neck {
    fn load(
        weights: &HashMap<String, Array>,
        prefix: &str,
        scale: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        let conv1_w = get_weight(weights, &format!("{prefix}.conv1.weight"))?;
        let c = conv1_w.shape()[0] as usize;
        let width = (c / scale) as i32;

        let mut branch_convs = Vec::new();
        let mut branch_biases = Vec::new();
        for k in 0..(scale - 1) {
            branch_convs.push(get_weight(weights, &format!("{prefix}.convs.{k}.weight"))?);
            branch_biases.push(weights.get(&format!("{prefix}.convs.{k}.bias")).cloned());
        }

        Ok(Self {
            conv1_w,
            conv1_b: weights.get(&format!("{prefix}.conv1.bias")).cloned(),
            bn1: BatchNorm1d::load(weights, &format!("{prefix}.bn1"))?,
            branch_convs,
            branch_biases,
            bn2: BatchNorm1d::load(weights, &format!("{prefix}.bn2"))?,
            conv3_w: get_weight(weights, &format!("{prefix}.conv3.weight"))?,
            conv3_b: weights.get(&format!("{prefix}.conv3.bias")).cloned(),
            bn3: BatchNorm1d::load(weights, &format!("{prefix}.bn3"))?,
            se: SeModule::load(weights, &format!("{prefix}.se"))?,
            scale,
            width,
            dilation,
            kernel,
        })
    }

    /// x: [B, L, C] → [B, L, C]
    fn forward(&self, x: &Array) -> Result<Array> {
        let w = self.width;

        // Pointwise in
        let h = conv1d_with_bias(x, &self.conv1_w, &self.conv1_b, 1, 0, 1)?;
        let h = self.bn1.forward(&h)?;
        let h = ops::relu(&h)?;

        // Split into `scale` chunks of width `w` along channel dim
        let mut out_chunks: Vec<Array> = Vec::with_capacity(self.scale);
        let mut prev: Option<Array> = None;
        let pad = compute_same_padding(self.kernel, self.dilation);

        for k in 0..self.scale {
            let chunk = h.slice_axes(&[2], &[k as i32 * w], &[(k + 1) as i32 * w])?;
            if k == 0 {
                out_chunks.push(chunk);
            } else {
                let branch_in = match prev.take() {
                    Some(p) => (&chunk + &p)?,
                    None => chunk,
                };
                let ck = conv1d_with_bias(
                    &branch_in,
                    &self.branch_convs[k - 1],
                    &self.branch_biases[k - 1],
                    1, pad, self.dilation,
                )?;
                prev = Some(ck.clone());
                out_chunks.push(ck);
            }
        }

        let refs: Vec<&Array> = out_chunks.iter().collect();
        let h = ops::concatenate(&refs, 2)?;
        let h = self.bn2.forward(&h)?;
        let h = ops::relu(&h)?;

        // Pointwise out
        let h = conv1d_with_bias(&h, &self.conv3_w, &self.conv3_b, 1, 0, 1)?;
        let h = self.bn3.forward(&h)?;
        let h = ops::relu(&h)?;

        // SE attention + residual
        let h = self.se.forward(&h)?;
        (x + &h)
    }
}

/// ECAPA-TDNN speaker encoder.
///
/// Input:  reference mel [1, T, n_mels]  (channels-last)
/// Output: speaker embedding [1, spk_dim]
pub struct EcapaTdnn {
    conv1_w: Array,
    conv1_b: Option<Array>,
    bn1: BatchNorm1d,
    layer1: Bottle2neck,
    layer2: Bottle2neck,
    layer3: Bottle2neck,
    layer4_w: Array,
    layer4_b: Option<Array>,
    attn_w1: Array,
    attn_b1: Option<Array>,
    attn_w2: Array,
    attn_b2: Option<Array>,
    bn5: BatchNorm1d,
    fc6_w: Array,
    fc6_b: Option<Array>,
    bn6: BatchNorm1d,
}

impl EcapaTdnn {
    pub fn load(weights: &HashMap<String, Array>) -> Result<Self> {
        let pfx = "bigvgan.speaker_encoder";
        Ok(Self {
            conv1_w: get_weight(weights, &format!("{pfx}.conv1.weight"))?,
            conv1_b: weights.get(&format!("{pfx}.conv1.bias")).cloned(),
            bn1: BatchNorm1d::load(weights, &format!("{pfx}.bn1"))?,
            layer1: Bottle2neck::load(weights, &format!("{pfx}.layer1"), 8, 3, 2)?,
            layer2: Bottle2neck::load(weights, &format!("{pfx}.layer2"), 8, 3, 3)?,
            layer3: Bottle2neck::load(weights, &format!("{pfx}.layer3"), 8, 3, 4)?,
            layer4_w: get_weight(weights, &format!("{pfx}.layer4.weight"))?,
            layer4_b: weights.get(&format!("{pfx}.layer4.bias")).cloned(),
            attn_w1: get_weight(weights, &format!("{pfx}.attention.0.weight"))?,
            attn_b1: weights.get(&format!("{pfx}.attention.0.bias")).cloned(),
            attn_w2: get_weight(weights, &format!("{pfx}.attention.4.weight"))?,
            attn_b2: weights.get(&format!("{pfx}.attention.4.bias")).cloned(),
            bn5: BatchNorm1d::load(weights, &format!("{pfx}.bn5"))?,
            fc6_w: get_weight(weights, &format!("{pfx}.fc6.weight"))?,
            fc6_b: weights.get(&format!("{pfx}.fc6.bias")).cloned(),
            bn6: BatchNorm1d::load(weights, &format!("{pfx}.bn6"))?,
        })
    }

    /// mel: [1, T, n_mels] (channels-last) → speaker embedding [1, spk_dim]
    pub fn forward(&self, mel: &Array) -> Result<Array> {
        // Initial TDNN: Conv1d(n_mels, C, kernel=5, pad=2)
        let h = conv1d_with_bias(mel, &self.conv1_w, &self.conv1_b, 1, 2, 1)?;
        let h = self.bn1.forward(&h)?;
        let h = ops::relu(&h)?;

        // 3 SE-Res2Net blocks
        let h1 = self.layer1.forward(&h)?;
        let h2 = self.layer2.forward(&h1)?;
        let h3 = self.layer3.forward(&h2)?;

        // Aggregate: concat then 1×1 conv
        let cat = ops::concatenate(&[&h1, &h2, &h3], 2)?;
        let h4 = conv1d_with_bias(&cat, &self.layer4_w, &self.layer4_b, 1, 0, 1)?;
        let h4 = ops::relu(&h4)?;

        // Attentive statistics pooling
        let mean_h4 = h4.mean_axis(1, true)?;
        let diff = (h4.clone() - &mean_h4)?;
        let var = (&diff * &diff)?.mean_axis(1, true)?;
        let std_h4 = ops::sqrt(&(&var + 1e-9f32)?)?;

        let mean_exp = mean_h4.broadcast_to(h4.shape())?;
        let std_exp = std_h4.broadcast_to(h4.shape())?;
        let attn_in = ops::concatenate(&[&h4, &mean_exp, &std_exp], 2)?;

        // Attention: two 1×1 convs with relu → tanh → softmax over L
        let e = conv1d_with_bias(&attn_in, &self.attn_w1, &self.attn_b1, 1, 0, 1)?;
        let e = ops::relu(&e)?;
        let e = ops::tanh(&e)?;
        let e = conv1d_with_bias(&e, &self.attn_w2, &self.attn_b2, 1, 0, 1)?;
        let attn = ops::softmax_axis(&e, 1, None::<bool>)?; // softmax over T dim

        // Weighted mean and std
        let w_mean = (&h4 * &attn)?.sum_axis(1, false)?;          // [B, C]
        let w_sq_sum = (&h4 * &h4)?.multiply(&attn)?.sum_axis(1, false)?;
        let w_std = ops::sqrt(&(&w_sq_sum - &(&w_mean * &w_mean)?)? + 1e-9f32)?;

        let pooled = ops::concatenate(&[&w_mean, &w_std], 1)?;    // [B, 2C]

        // FC6 + BN6 — expand/squeeze [B, 2C] ↔ [B, 1, 2C]
        let pooled = pooled.reshape(&[pooled.shape()[0], 1, pooled.shape()[1]])?;
        let pooled = self.bn5.forward(&pooled)?;
        let emb = apply_linear_bias(ops::matmul(&pooled, &self.fc6_w.t())?, &self.fc6_b)?;
        let emb = self.bn6.forward(&emb)?;

        // [B, 1, spk_dim] → [B, spk_dim]
        emb.squeeze(1)
    }
}

// ─── MRF (Multi-Receptive Field) residual block ───────────────────────────────

struct ResBlockLayer {
    conv1_w: Array,
    conv1_b: Option<Array>,
    act1_alpha: Array,
    act1_beta: Array,
    conv2_w: Array,
    conv2_b: Option<Array>,
    act2_alpha: Array,
    act2_beta: Array,
    dilation: usize,
    kernel: usize,
}

/// Single MRF ResBlock: paired (SnakeBeta + dilated conv) layers.
struct ResBlock {
    layers: Vec<ResBlockLayer>,
}

impl ResBlock {
    fn load(
        weights: &HashMap<String, Array>,
        prefix: &str,
        kernel: usize,
        dilations: &[usize],
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for (j, &dil) in dilations.iter().enumerate() {
            layers.push(ResBlockLayer {
                conv1_w: get_weight(weights, &format!("{prefix}.convs1.{j}.weight"))?,
                conv1_b: weights.get(&format!("{prefix}.convs1.{j}.bias")).cloned(),
                act1_alpha: get_weight(weights, &format!("{prefix}.activations1.{j}.alpha"))?,
                act1_beta: get_weight(weights, &format!("{prefix}.activations1.{j}.beta"))?,
                conv2_w: get_weight(weights, &format!("{prefix}.convs2.{j}.weight"))?,
                conv2_b: weights.get(&format!("{prefix}.convs2.{j}.bias")).cloned(),
                act2_alpha: get_weight(weights, &format!("{prefix}.activations2.{j}.alpha"))?,
                act2_beta: get_weight(weights, &format!("{prefix}.activations2.{j}.beta"))?,
                dilation: dil,
                kernel,
            });
        }
        Ok(Self { layers })
    }

    /// x: [B, L, C] → [B, L, C]
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = x.clone();
        for layer in &self.layers {
            let pad = compute_same_padding(layer.kernel, layer.dilation);
            let h = snake_beta(&out, &layer.act1_alpha, &layer.act1_beta)?;
            let h = conv1d_with_bias(&h, &layer.conv1_w, &layer.conv1_b, 1, pad, layer.dilation)?;
            let h = snake_beta(&h, &layer.act2_alpha, &layer.act2_beta)?;
            let h = conv1d_with_bias(&h, &layer.conv2_w, &layer.conv2_b, 1, pad, 1)?;
            out = (&out + &h)?;
        }
        Ok(out)
    }
}

// ─── Upsampling stage ─────────────────────────────────────────────────────────

struct UpStage {
    ups_w: Array,          // ConvTranspose1d weights [C_out, kernel, C_in] (MLX format)
    ups_b: Option<Array>,
    cond: nn::Linear,      // FiLM conditioning: spk_dim → 2*ch_out
    resblocks: Vec<ResBlock>,
    ch_out: i32,
    stride: usize,
    kernel: usize,
}

impl UpStage {
    fn load(
        weights: &HashMap<String, Array>,
        stage_idx: usize,
        rb_start: usize,
        n_resblocks: usize,
        resblock_kernels: &[usize],
        resblock_dilations: &[Vec<usize>],
        stride: usize,
        kernel: usize,
    ) -> Result<Self> {
        let pfx = "bigvgan";
        let ups_w = get_weight(weights, &format!("{pfx}.ups.{stage_idx}.weight"))?;
        let ups_b = weights.get(&format!("{pfx}.ups.{stage_idx}.bias")).cloned();
        // ConvTranspose1d weight shape [C_out, kernel, C_in] in MLX format
        let ch_out = ups_w.shape()[0];

        let cond = load_linear(weights, &format!("{pfx}.conds.{stage_idx}"))?;

        let mut resblocks = Vec::new();
        for k in 0..n_resblocks {
            let rb_idx = rb_start + k;
            let rk = resblock_kernels[k % resblock_kernels.len()];
            let dils = &resblock_dilations[k % resblock_dilations.len()];
            resblocks.push(ResBlock::load(
                weights,
                &format!("{pfx}.resblocks.{rb_idx}"),
                rk,
                dils,
            )?);
        }

        Ok(Self { ups_w, ups_b, cond, resblocks, ch_out, stride, kernel })
    }

    /// x:     [B, L, C_in]
    /// spk:   [B, spk_dim]
    /// → out: [B, L*stride, C_out]
    fn forward(&self, x: &Array, spk: &Array) -> Result<Array> {
        // Transposed convolution (upsampling)
        let h = ops::conv_transpose1d(
            x, &self.ups_w,
            self.stride as i32, 0i32, 1i32, 0i32, 1i32,
        )?;
        let h = apply_conv_bias(h, &self.ups_b)?;

        // Trim padding artifacts: (kernel - stride) / 2 on each side
        let trim = (self.kernel - self.stride) / 2;
        let l_out = h.shape()[1];
        let h = if trim > 0 {
            let start = trim as i32;
            let end = l_out - trim as i32;
            h.slice_axes(&[1], &[start], &[end])?
        } else {
            h
        };

        // FiLM conditioning: spk → [B, 2*C_out] → scale, shift
        let cond_out = self.cond.forward(spk)?; // [B, 2*C_out]
        let cond_out = cond_out.reshape(&[cond_out.shape()[0], 1, cond_out.shape()[1]])?;
        let c = self.ch_out;
        let scale = (cond_out.slice_axes(&[2], &[0], &[c])? + array!(1.0f32))?;
        let shift = cond_out.slice_axes(&[2], &[c], &[2 * c])?;
        let h = ((&h * &scale)? + &shift)?;

        // MRF: sum over all ResBlocks, then scale by 1/n
        let mut mrf_out: Option<Array> = None;
        for rb in &self.resblocks {
            let rb_out = rb.forward(&h)?;
            mrf_out = Some(match mrf_out {
                None => rb_out,
                Some(acc) => (&acc + &rb_out)?,
            });
        }
        let h = mrf_out.ok_or_else(|| Error::Model("no resblocks in stage".into()))?;
        let n = self.resblocks.len() as f32;
        Ok(h * (1.0f32 / n))
    }
}

// ─── BigVGAN ──────────────────────────────────────────────────────────────────

/// BigVGAN vocoder.
pub struct BigVgan {
    conv_pre_w: Array,
    conv_pre_b: Option<Array>,
    speaker_encoder: EcapaTdnn,
    cond_layer: nn::Linear,
    stages: Vec<UpStage>,
    post_alpha: Array,
    post_beta: Array,
    conv_post_w: Array,
    conv_post_b: Option<Array>,
}

impl BigVgan {
    pub fn load(weights: &HashMap<String, Array>, cfg: &BigvganConfig) -> Result<Self> {
        let pfx = "bigvgan";
        let n_stages = cfg.upsample_rates.len();
        let n_rk = cfg.resblock_kernel_sizes.len();

        let conv_pre_w = get_weight(weights, &format!("{pfx}.conv_pre.weight"))?;
        let conv_pre_b = weights.get(&format!("{pfx}.conv_pre.bias")).cloned();

        let speaker_encoder = EcapaTdnn::load(weights)?;

        let cond_layer = load_linear(weights, &format!("{pfx}.cond_layer"))?;

        let mut stages = Vec::with_capacity(n_stages);
        let mut rb_global = 0usize;
        for i in 0..n_stages {
            stages.push(UpStage::load(
                weights, i, rb_global, n_rk,
                &cfg.resblock_kernel_sizes,
                &cfg.resblock_dilation_sizes,
                cfg.upsample_rates[i],
                cfg.upsample_kernel_sizes[i],
            )?);
            rb_global += n_rk;
        }

        let post_alpha = get_weight(weights, &format!("{pfx}.activation_post.alpha"))?;
        let post_beta = get_weight(weights, &format!("{pfx}.activation_post.beta"))?;
        let conv_post_w = get_weight(weights, &format!("{pfx}.conv_post.weight"))?;
        let conv_post_b = weights.get(&format!("{pfx}.conv_post.bias")).cloned();

        Ok(Self {
            conv_pre_w, conv_pre_b,
            speaker_encoder, cond_layer, stages,
            post_alpha, post_beta, conv_post_w, conv_post_b,
        })
    }

    /// Synthesize waveform from GPT latent and reference mel.
    ///
    /// gpt_latent:    [1, T, gpt_dim]   — GPT hidden states (channels-last)
    /// reference_mel: [1, T_ref, n_mels] — reference mel (channels-last)
    ///
    /// Returns waveform [T_wav] (mono, flat).
    pub fn forward(&self, gpt_latent: &Array, reference_mel: &Array) -> Result<Array> {
        // 1. Speaker embedding: [1, T_ref, n_mels] → [1, spk_dim]
        let spk = self.speaker_encoder.forward(reference_mel)?;

        // 2. Initial conv: gpt_latent → [1, T, ch_init]
        let pad_pre = compute_same_padding(self.conv_pre_w.shape()[1] as usize, 1);
        let h = conv1d_with_bias(gpt_latent, &self.conv_pre_w, &self.conv_pre_b, 1, pad_pre, 1)?;

        // 3. Global FiLM conditioning
        let cond_out = self.cond_layer.forward(&spk)?;
        let ch = h.shape()[2];
        let cond_out = cond_out.reshape(&[1, 1, cond_out.shape()[1]])?;
        let scale = (cond_out.slice_axes(&[2], &[0], &[ch])? + array!(1.0f32))?;
        let shift = cond_out.slice_axes(&[2], &[ch], &[2 * ch])?;
        let h = ((&h * &scale)? + &shift)?;

        // 4. Upsampling stages
        let mut h = h;
        for stage in &self.stages {
            h = stage.forward(&h, &spk)?;
        }

        // 5. Post-activation + final conv + tanh
        let h = snake_beta(&h, &self.post_alpha, &self.post_beta)?;
        let pad_post = compute_same_padding(self.conv_post_w.shape()[1] as usize, 1);
        let h = conv1d_with_bias(&h, &self.conv_post_w, &self.conv_post_b, 1, pad_post, 1)?;
        let h = ops::tanh(&h)?;

        // Flatten: [1, T_wav, 1] → [T_wav]
        let t_wav = h.shape()[1];
        h.reshape(&[t_wav])
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Symmetric "same" padding for Conv1d with given kernel and dilation.
fn compute_same_padding(kernel: usize, dilation: usize) -> usize {
    dilation * (kernel - 1) / 2
}

/// Conv1d with optional bias: input [B, L, C_in], weight [C_out, kernel, C_in].
fn conv1d_with_bias(
    x: &Array,
    w: &Array,
    bias: &Option<Array>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Array> {
    let out = ops::conv1d(x, w, stride as i32, padding as i32, dilation as i32, 1i32)?;
    apply_conv_bias(out, bias)
}

/// Add optional bias [C_out] to a conv output [B, L, C_out].
fn apply_conv_bias(x: Array, bias: &Option<Array>) -> Result<Array> {
    match bias {
        Some(b) => {
            let c = b.shape()[0];
            (x + b.reshape(&[1, 1, c])?)
        }
        None => Ok(x),
    }
}

/// Apply an optional Linear bias as [1, 1, C] for matmul outputs [B, 1, C].
fn apply_linear_bias(x: Array, bias: &Option<Array>) -> Result<Array> {
    match bias {
        Some(b) => {
            let c = b.shape()[0];
            (x + b.reshape(&[1, 1, c])?)
        }
        None => Ok(x),
    }
}
