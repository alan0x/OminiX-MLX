//! Audio conditioner: Conformer encoder + Perceiver resampler.
//!
//! Processes reference mel spectrogram into conditioning vectors for GPT.
//!
//! Key prefixes in model.safetensors:
//!   conditioning_encoder.*   — Conformer (Conv2dSubsampling + blocks)
//!   perceiver_encoder.*      — PerceiverResampler

use std::collections::HashMap;

use mlx_rs::{
    nn,
    ops::{self, matmul},
    Array,
};

use crate::config::ConformerPerceiverConfig;
use crate::error::{Error, Result};

// ─── Conformer ────────────────────────────────────────────────────────────────

/// Conformer feed-forward module (pre-LN, 0.5 residual weight).
struct FeedForward {
    norm: nn::LayerNorm,
    w1: nn::Linear,
    w2: nn::Linear,
}

impl FeedForward {
    fn load(weights: &HashMap<String, Array>, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: load_layer_norm(weights, &format!("{prefix}.norm_ff"))?,
            w1: load_linear(weights, &format!("{prefix}.feed_forward.w_1"))?,
            w2: load_linear(weights, &format!("{prefix}.feed_forward.w_2"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.norm.forward(x)?;
        let h = self.w1.forward(&h)?;
        let h = ops::sigmoid(&h)? * &h; // SiLU
        let h = self.w2.forward(&h)?;
        Ok((x + &h * 0.5f32)?)
    }
}

/// Single Conformer block.
struct ConformerBlock {
    ff1: FeedForward,
    ff2: FeedForward,
    norm_mha: nn::LayerNorm,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    n_heads: usize,
    norm_conv: nn::LayerNorm,
    conv_pointwise1: nn::Linear,
    conv_depthwise: nn::Linear, // approximated as linear on last dim
    conv_pointwise2: nn::Linear,
    norm_final: nn::LayerNorm,
}

impl ConformerBlock {
    fn load(weights: &HashMap<String, Array>, prefix: &str) -> Result<Self> {
        Ok(Self {
            ff1: FeedForward::load(weights, &format!("{prefix}.feed_forward_macaron"))?,
            ff2: FeedForward::load(weights, &format!("{prefix}.feed_forward"))?,
            norm_mha: load_layer_norm(weights, &format!("{prefix}.norm_mha"))?,
            q_proj: load_linear(weights, &format!("{prefix}.self_attn.linear_q"))?,
            k_proj: load_linear(weights, &format!("{prefix}.self_attn.linear_k"))?,
            v_proj: load_linear(weights, &format!("{prefix}.self_attn.linear_v"))?,
            out_proj: load_linear(weights, &format!("{prefix}.self_attn.linear_out"))?,
            n_heads: 8,
            norm_conv: load_layer_norm(weights, &format!("{prefix}.norm_conv"))?,
            conv_pointwise1: load_linear(
                weights,
                &format!("{prefix}.conv_module.pointwise_conv1"),
            )?,
            conv_depthwise: load_linear(
                weights,
                &format!("{prefix}.conv_module.depthwise_conv"),
            )?,
            conv_pointwise2: load_linear(
                weights,
                &format!("{prefix}.conv_module.pointwise_conv2"),
            )?,
            norm_final: load_layer_norm(weights, &format!("{prefix}.norm_final"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        // FF macaron
        let x = self.ff1.forward(x)?;

        // Multi-head self-attention
        let normed = self.norm_mha.forward(&x)?;
        let shape = normed.shape();
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        let head_dim = d / self.n_heads as i32;

        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;

        let q = q.reshape(&[b, t, self.n_heads as i32, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, t, self.n_heads as i32, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, t, self.n_heads as i32, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;

        let scale = (head_dim as f32).sqrt().recip();
        let scores = matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)? * scale;
        let attn = ops::softmax(&scores, -1)?;
        let out = matmul(&attn, &v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, t, d])?;
        let attn_out = self.out_proj.forward(&out)?;
        let x = (x + attn_out)?;

        // Convolution module
        let normed = self.norm_conv.forward(&x)?;
        let h = self.conv_pointwise1.forward(&normed)?;
        // GLU: split in half, sigmoid gate
        let half = h.shape()[2] / 2;
        let h1 = h.slice_axes(&[2], &[0], &[half])?;
        let h2 = h.slice_axes(&[2], &[half], &[h.shape()[2]])?;
        let h = (h1 * ops::sigmoid(&h2)?)?;
        let h = self.conv_depthwise.forward(&h)?;
        let h = ops::gelu(&h)?;
        let h = self.conv_pointwise2.forward(&h)?;
        let x = (x + h)?;

        // FF final
        let x = self.ff2.forward(&x)?;
        let x = self.norm_final.forward(&x)?;

        Ok(x)
    }
}

/// Conformer encoder (Conv2dSubsampling + N blocks).
///
/// Weights prefix: `conditioning_encoder`
pub struct ConformerEncoder {
    /// Conv2d subsampling: projects [B, 1, T, n_mels] → [B, T', d_model]
    conv_sub_conv1: Array, // [d_model, 1, 3, 3] weights
    conv_sub_conv2: Array,
    conv_sub_linear: nn::Linear,
    blocks: Vec<ConformerBlock>,
    d_model: usize,
    n_mels: usize,
}

impl ConformerEncoder {
    pub fn load(
        weights: &HashMap<String, Array>,
        cfg: &ConformerPerceiverConfig,
        n_mels: usize,
    ) -> Result<Self> {
        let pfx = "conditioning_encoder";

        // Conv2dSubsampling weights (embed)
        let c1 = get_weight(weights, &format!("{pfx}.embed.conv.0.weight"))?;
        let c2 = get_weight(weights, &format!("{pfx}.embed.conv.2.weight"))?;
        let linear = load_linear(weights, &format!("{pfx}.embed.out.0"))?;

        let mut blocks = Vec::with_capacity(cfg.num_blocks);
        for i in 0..cfg.num_blocks {
            let block = ConformerBlock::load(weights, &format!("{pfx}.encoders.{i}"))?;
            blocks.push(block);
        }

        Ok(Self {
            conv_sub_conv1: c1,
            conv_sub_conv2: c2,
            conv_sub_linear: linear,
            blocks,
            d_model: cfg.output_size,
            n_mels,
        })
    }

    /// Forward pass.
    ///
    /// Input:  mel `[1, n_mels, T]`
    /// Output: `[1, T', d_model]`
    pub fn forward(&self, mel: &Array) -> Result<Array> {
        // Reshape mel [B, n_mels, T] → [B, T, n_mels, 1] (MLX conv2d: [N, H, W, C_in])
        let shape = mel.shape();
        let (b, n_mels, t) = (shape[0], shape[1], shape[2]);
        let x = mel.transpose_axes(&[0, 2, 1])? // [B, T, n_mels]
            .reshape(&[b, t, n_mels, 1])?;      // [B, T, n_mels, 1]

        // Conv2dSubsampling: 2 × Conv2d(stride=(2,2)) → flatten → linear
        // Output after each conv: [B, T/2, n_mels/2, C_out]
        let x = ops::conv2d(&x, &self.conv_sub_conv1, (2, 2), (0, 0), (1, 1), 1)?;
        let x = ops::relu(&x)?;
        let x = ops::conv2d(&x, &self.conv_sub_conv2, (2, 2), (0, 0), (1, 1), 1)?;
        let x = ops::relu(&x)?;
        // Flatten freq+channel dims: [B, T', F', C] → [B, T', F'*C]
        let s = x.shape();
        let x = x.reshape(&[s[0], s[1], s[2] * s[3]])?;
        let x = self.conv_sub_linear.forward(&x)?;

        // Conformer blocks
        let mut h = x;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }

        Ok(h)
    }
}

// ─── Perceiver Resampler ──────────────────────────────────────────────────────

struct PerceiverLayer {
    attn_q: nn::Linear,
    attn_k: nn::Linear,
    attn_v: nn::Linear,
    attn_out: nn::Linear,
    ff_w1: nn::Linear,
    ff_w2: nn::Linear,
    n_heads: usize,
}

impl PerceiverLayer {
    fn load(weights: &HashMap<String, Array>, prefix: &str, n_heads: usize) -> Result<Self> {
        Ok(Self {
            attn_q: load_linear(weights, &format!("{prefix}.0.query"))?,
            attn_k: load_linear(weights, &format!("{prefix}.0.key"))?,
            attn_v: load_linear(weights, &format!("{prefix}.0.value"))?,
            attn_out: load_linear(weights, &format!("{prefix}.0.out_proj"))?,
            ff_w1: load_linear(weights, &format!("{prefix}.1.net.0"))?,
            ff_w2: load_linear(weights, &format!("{prefix}.1.net.2"))?,
            n_heads,
        })
    }

    fn forward(&self, latents: &Array, context: &Array) -> Result<Array> {
        let shape = latents.shape();
        let (b, n_lat, d) = (shape[0], shape[1], shape[2]);
        let head_dim = d / self.n_heads as i32;

        // kv = concat(context, latents) along sequence dim
        let kv = ops::concatenate(&[context, latents], 1)?;

        let q = self.attn_q.forward(latents)?
            .reshape(&[b, n_lat, self.n_heads as i32, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self.attn_k.forward(&kv)?
            .reshape(&[b, kv.shape()[1], self.n_heads as i32, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self.attn_v.forward(&kv)?
            .reshape(&[b, kv.shape()[1], self.n_heads as i32, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let scale = (head_dim as f32).sqrt().recip();
        let attn = ops::softmax(&(matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)? * scale), -1)?;
        let out = matmul(&attn, &v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n_lat, d])?;
        let latents = (latents + self.attn_out.forward(&out)?)?;

        // Feed-forward
        let h = self.ff_w1.forward(&latents)?;
        let h = ops::gelu(&h)?;
        let h = self.ff_w2.forward(&h)?;
        Ok((latents + h)?)
    }
}

/// Perceiver resampler.
///
/// Compresses variable-length Conformer output to fixed N latents.
/// Weights prefix: `perceiver_encoder`
pub struct PerceiverResampler {
    proj_context: nn::Linear,
    latents: Array,         // [n_latents, d_model] learnable
    layers: Vec<PerceiverLayer>,
    norm: nn::RmsNorm,
    n_heads: usize,
}

impl PerceiverResampler {
    pub fn load(
        weights: &HashMap<String, Array>,
        cfg: &ConformerPerceiverConfig,
    ) -> Result<Self> {
        let pfx = "perceiver_encoder";
        let n_depth = 2; // standard perceiver depth
        let n_heads = cfg.attention_heads;

        let proj = load_linear(weights, &format!("{pfx}.proj_context"))?;
        let latents = get_weight(weights, &format!("{pfx}.latents"))?;

        let mut layers = Vec::with_capacity(n_depth);
        for i in 0..n_depth {
            layers.push(PerceiverLayer::load(
                weights,
                &format!("{pfx}.layers.{i}"),
                n_heads,
            )?);
        }

        let norm = load_rms_norm(weights, &format!("{pfx}.norm"))?;

        Ok(Self { proj_context: proj, latents, layers, norm, n_heads })
    }

    /// Forward.
    ///
    /// Input:  context `[B, T', d_enc]`
    /// Output: `[B, n_latents, d_model]`
    pub fn forward(&self, context: &Array) -> Result<Array> {
        let b = context.shape()[0];
        let ctx = self.proj_context.forward(context)?;

        // Broadcast learnable latents to batch
        let n_lat = self.latents.shape()[0];
        let d = self.latents.shape()[1];
        let lat = self.latents
            .reshape(&[1, n_lat, d])?
            .broadcast_to(&[b, n_lat, d])?;

        let mut lat = lat;
        for layer in &self.layers {
            lat = layer.forward(&lat, &ctx)?;
        }

        let out = self.norm.forward(&lat)?;
        Ok(out)
    }
}

// ─── Weight loading helpers ───────────────────────────────────────────────────

pub fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

pub fn load_linear(weights: &HashMap<String, Array>, prefix: &str) -> Result<nn::Linear> {
    let w = get_weight(weights, &format!("{prefix}.weight"))?;
    let b = weights.get(&format!("{prefix}.bias")).cloned();
    Ok(nn::Linear {
        weight: mlx_rs::module::Param::new(w),
        bias: b.map(mlx_rs::module::Param::new),
    })
}

pub fn load_layer_norm(weights: &HashMap<String, Array>, prefix: &str) -> Result<nn::LayerNorm> {
    let w = get_weight(weights, &format!("{prefix}.weight"))?;
    let b = get_weight(weights, &format!("{prefix}.bias"))?;
    let eps = 1e-5;
    Ok(nn::LayerNorm::new_with_weight_bias(w.shape()[0] as usize, eps, w, b)
        .map_err(|e| Error::Model(format!("LayerNorm init: {e}")))?)
}

pub fn load_rms_norm(weights: &HashMap<String, Array>, prefix: &str) -> Result<nn::RmsNorm> {
    let w = get_weight(weights, &format!("{prefix}.weight"))?;
    let eps = 1e-6;
    Ok(nn::RmsNorm::new_with_weight(w.shape()[0] as usize, eps, w)
        .map_err(|e| Error::Model(format!("RmsNorm init: {e}")))?)
}
