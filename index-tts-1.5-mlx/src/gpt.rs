//! GPT-2 style autoregressive decoder for IndexTTS 1.5.
//!
//! Key prefixes:
//!   text_embedding.*       — top-level text token embedding
//!   mel_embedding.*        — top-level mel code embedding
//!   mel_pos_embedding.*    — positional encoding for mel
//!   text_pos_embedding.*   — positional encoding for text
//!   text_head.*            — text output head (not used in inference)
//!   mel_head.*             — mel output head (logits over 8194 tokens)
//!   final_norm.*           — LayerNorm before mel_head
//!   gpt.h.{i}.*            — 24 transformer blocks
//!   gpt.ln_f.*             — final LayerNorm inside GPT
//!
//! Note: gpt.wte and gpt.wpe are patched to Identity in mlx-audio,
//!       so they are NOT present in the safetensors file.

use std::collections::HashMap;

use mlx_rs::{
    array,
    nn,
    ops::{self, matmul},
    Array,
};
use mlx_rs_core::cache::{KVCache, KeyValueCache};

use crate::conditioner::{get_weight, load_layer_norm, load_linear};
use crate::error::{Error, Result};

// ─── Transformer block ────────────────────────────────────────────────────────

struct TransformerBlock {
    ln_1: nn::LayerNorm,
    // Combined QKV: weight [3*d, d], bias [3*d]
    c_attn_w: Array,
    c_attn_b: Array,
    c_proj: nn::Linear,
    ln_2: nn::LayerNorm,
    mlp_fc: nn::Linear,
    mlp_proj: nn::Linear,
    n_heads: usize,
    head_dim: usize,
}

impl TransformerBlock {
    fn load(weights: &HashMap<String, Array>, prefix: &str, n_heads: usize) -> Result<Self> {
        let d_model = get_weight(weights, &format!("{prefix}.ln_1.weight"))?.shape()[0] as usize;
        let head_dim = d_model / n_heads;

        Ok(Self {
            ln_1: load_layer_norm(weights, &format!("{prefix}.ln_1"))?,
            c_attn_w: get_weight(weights, &format!("{prefix}.attn.c_attn.weight"))?,
            c_attn_b: get_weight(weights, &format!("{prefix}.attn.c_attn.bias"))?,
            c_proj: load_linear(weights, &format!("{prefix}.attn.c_proj"))?,
            ln_2: load_layer_norm(weights, &format!("{prefix}.ln_2"))?,
            mlp_fc: load_linear(weights, &format!("{prefix}.mlp.c_fc"))?,
            mlp_proj: load_linear(weights, &format!("{prefix}.mlp.c_proj"))?,
            n_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Array, cache: &mut KVCache) -> Result<Array> {
        let shape = x.shape();
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        let h = self.n_heads as i32;
        let hd = self.head_dim as i32;

        // Pre-LN attention
        let normed = self.ln_1.forward(x)?;

        // Combined QKV projection
        let qkv = (matmul(&normed, &self.c_attn_w.t())? + &self.c_attn_b)?;
        // Split: [B, T, 3*d] → q, k, v each [B, T, d]
        let q = qkv.slice_axes(&[2], &[0], &[d])?;
        let k = qkv.slice_axes(&[2], &[d], &[2 * d])?;
        let v = qkv.slice_axes(&[2], &[2 * d], &[3 * d])?;

        let q = q.reshape(&[b, t, h, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, t, h, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, t, h, hd])?.transpose_axes(&[0, 2, 1, 3])?;

        // KV cache update
        let (k, v) = cache.update_and_fetch(k, v)?;

        let scale = (hd as f32).sqrt().recip();
        let seq_len = k.shape()[2];
        let scores = matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)? * scale;
        // Causal mask for prefill (t > 1)
        let scores = if t > 1 {
            let mask = ops::triu(
                &ops::full::<f32>(&[t, seq_len], array!(f32::NEG_INFINITY))?,
                1,
            )?;
            (scores + mask.reshape(&[1, 1, t, seq_len])?)?
        } else {
            scores
        };
        let attn = ops::softmax(&scores, -1)?;
        let out = matmul(&attn, &v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, t, d])?;
        let attn_out = self.c_proj.forward(&out)?;
        let x = (x + attn_out)?;

        // Pre-LN MLP
        let normed = self.ln_2.forward(&x)?;
        let h_mlp = self.mlp_fc.forward(&normed)?;
        let h_mlp = ops::gelu(&h_mlp)?;
        let h_mlp = self.mlp_proj.forward(&h_mlp)?;
        Ok((x + h_mlp)?)
    }
}

// ─── GPT model ────────────────────────────────────────────────────────────────

pub struct Gpt {
    // Top-level embeddings
    text_embedding: nn::Embedding,
    mel_embedding: nn::Embedding,
    mel_pos_embedding: nn::Embedding,
    text_pos_embedding: nn::Embedding,
    // GPT transformer
    blocks: Vec<TransformerBlock>,
    ln_f: nn::LayerNorm,
    // Output heads
    mel_head: nn::Linear,
    final_norm: nn::LayerNorm,
    // Config
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_model: usize,
    pub start_mel_token: u32,
    pub stop_mel_token: u32,
    pub start_text_token: u32,
    pub stop_text_token: u32,
    pub number_mel_codes: usize,
}

impl Gpt {
    pub fn load(weights: &HashMap<String, Array>, config: &crate::config::GptConfig) -> Result<Self> {
        let n_layers = config.layers;
        let n_heads = config.heads;
        let d_model = config.model_dim;

        // Embeddings
        let text_emb_w = get_weight(weights, "text_embedding.weight")?;
        let mel_emb_w = get_weight(weights, "mel_embedding.weight")?;
        let mel_pos_w = get_weight(weights, "mel_pos_embedding.weight")?;
        let text_pos_w = get_weight(weights, "text_pos_embedding.weight")?;

        let text_embedding = nn::Embedding {
            weight: mlx_rs::module::Param::new(text_emb_w),
        };
        let mel_embedding = nn::Embedding {
            weight: mlx_rs::module::Param::new(mel_emb_w),
        };
        let mel_pos_embedding = nn::Embedding {
            weight: mlx_rs::module::Param::new(mel_pos_w),
        };
        let text_pos_embedding = nn::Embedding {
            weight: mlx_rs::module::Param::new(text_pos_w),
        };

        // GPT transformer blocks
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            blocks.push(TransformerBlock::load(
                weights,
                &format!("gpt.h.{i}"),
                n_heads,
            )?);
        }

        let ln_f = load_layer_norm(weights, "gpt.ln_f")?;
        let final_norm = load_layer_norm(weights, "final_norm")?;
        let mel_head = load_linear(weights, "mel_head")?;

        Ok(Self {
            text_embedding,
            mel_embedding,
            mel_pos_embedding,
            text_pos_embedding,
            blocks,
            ln_f,
            mel_head,
            final_norm,
            n_layers,
            n_heads,
            d_model,
            start_mel_token: config.start_mel_token as u32,
            stop_mel_token: config.stop_mel_token as u32,
            start_text_token: config.start_text_token as u32,
            stop_text_token: config.stop_text_token as u32,
            number_mel_codes: config.number_mel_codes,
        })
    }

    /// Build input embeddings for prefill.
    ///
    /// Layout: [conditioning_tokens | text_tokens | mel_bos]
    /// conditioning: [B, N_cond, d_model] from perceiver
    /// text_ids:     [B, L_text]
    pub fn build_prefill_embeds(
        &self,
        text_ids: &Array,
        conditioning: &Array,
    ) -> Result<Array> {
        let b = text_ids.shape()[0];
        let l = text_ids.shape()[1];

        // Text embeddings + positional
        let text_emb = self.text_embedding.forward(text_ids)?;
        let text_pos = self.text_pos_embedding.forward(
            &Array::from_slice(&(0..l as u32).collect::<Vec<_>>(), &[1, l]),
        )?
        .broadcast_to(&[b, l, self.d_model as i32])?;
        let text_emb = (text_emb + text_pos)?;

        // BOS mel token
        let bos = Array::from_slice(&[self.start_mel_token], &[1, 1]);
        let bos_emb = self.mel_embedding.forward(&bos.broadcast_to(&[b, 1])?)?;

        // Concatenate: [conditioning | text | mel_bos]
        let prefill = ops::concatenate(&[conditioning, &text_emb, &bos_emb], 1)?;
        Ok(prefill)
    }

    /// Single-step forward for one token (decode mode).
    ///
    /// Returns `(logits, hidden_last)`:
    ///   - `logits`:      `[B, number_mel_codes]` — mel code distribution
    ///   - `hidden_last`: `[B, d_model]` — last-position hidden state (for BigVGAN)
    pub fn forward_step(
        &self,
        embeds: &Array,
        caches: &mut Vec<KVCache>,
    ) -> Result<(Array, Array)> {
        let mut h = embeds.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &mut caches[i])?;
        }
        h = self.ln_f.forward(&h)?;
        h = self.final_norm.forward(&h)?;
        // Take last position hidden state for BigVGAN latent
        let t = h.shape()[1];
        let hidden_last = h.slice_axes(&[1], &[t - 1], &[t])?.squeeze(1)?; // [B, d_model]
        let logits = self.mel_head.forward(&h)?;
        let logits = logits.slice_axes(&[1], &[t - 1], &[t])?.squeeze(1)?;
        Ok((logits, hidden_last))
    }

    /// Autoregressive mel code generation.
    ///
    /// Returns generated mel codes (excluding BOS/EOS), and final hidden states
    /// for BigVGAN.
    pub fn generate(
        &self,
        text_ids: &Array,
        conditioning: &Array,
        max_mel_tokens: usize,
        top_k: usize,
        top_p: f32,
        temperature: f32,
    ) -> Result<(Vec<u32>, Array)> {
        let b = text_ids.shape()[0];
        assert_eq!(b, 1, "batch size must be 1 for autoregressive generation");

        // Initialize KV caches
        let mut caches: Vec<KVCache> = (0..self.n_layers)
            .map(|_| KVCache::new(self.n_heads, self.d_model / self.n_heads))
            .collect();

        // Prefill
        let prefill_emb = self.build_prefill_embeds(text_ids, conditioning)?;
        let _ = self.forward_step(&prefill_emb, &mut caches)?;

        // Autoregressive decode — collect GPT hidden states for BigVGAN
        let mut generated: Vec<u32> = Vec::new();
        let mut last_token = self.start_mel_token;
        let mut all_hidden: Vec<Array> = Vec::new(); // [B, d_model] per step

        for _ in 0..max_mel_tokens {
            let mel_pos = generated.len() as u32;
            let tok_arr = Array::from_slice(&[last_token], &[1, 1]);
            let tok_emb = self.mel_embedding.forward(&tok_arr)?;
            let pos_arr = Array::from_slice(&[mel_pos], &[1, 1]);
            let pos_emb = self.mel_pos_embedding.forward(&pos_arr)?;
            let step_emb = (tok_emb + pos_emb)?;

            let (logits, hidden) = self.forward_step(&step_emb, &mut caches)?;
            // hidden: [B, d_model] → store as [B, 1, d_model]
            all_hidden.push(hidden.reshape(&[1, 1, self.d_model as i32])?);

            let next_token = crate::sampling::sample(
                &logits, temperature, top_k, top_p,
            )?;

            if next_token == self.stop_mel_token {
                break;
            }
            generated.push(next_token);
            last_token = next_token;
        }

        // Stack hidden states: [1, T, d_model] for BigVGAN
        let latent = if all_hidden.is_empty() {
            Array::zeros::<f32>(&[1, 1, self.d_model as i32])?
        } else {
            ops::concatenate(&all_hidden.iter().collect::<Vec<_>>(), 1)?
        };

        Ok((generated, latent))
    }
}
