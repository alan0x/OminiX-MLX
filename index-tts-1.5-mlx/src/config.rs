//! IndexTTS 1.5 config.json deserialization.
//!
//! Matches the structure of mlx-community/IndexTTS-1.5 config.json.

use serde::Deserialize;
use std::path::Path;

use crate::error::{Error, Result};

#[derive(Debug, Deserialize)]
pub struct IndexTtsConfig {
    pub model_type: String,
    pub version: f32,
    pub gpt: GptConfig,
    pub vqvae: VqvaeConfig,
    pub bigvgan: BigvganConfig,
    pub dataset: DatasetConfig,
    pub gpt_checkpoint: Option<String>,
    pub dvae_checkpoint: Option<String>,
    pub bigvgan_checkpoint: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GptConfig {
    pub model_dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub max_mel_tokens: usize,
    pub max_text_tokens: usize,
    pub number_text_tokens: usize,
    pub number_mel_codes: usize,
    pub start_mel_token: usize,
    pub stop_mel_token: usize,
    pub start_text_token: usize,
    pub stop_text_token: usize,
    pub mel_length_compression: usize,
    pub use_mel_codes_as_input: bool,
    pub condition_type: String,
    pub condition_module: ConformerPerceiverConfig,
}

#[derive(Debug, Deserialize)]
pub struct ConformerPerceiverConfig {
    pub output_size: usize,
    pub linear_units: usize,
    pub attention_heads: usize,
    pub num_blocks: usize,
    pub input_layer: String,
    pub perceiver_mult: usize,
}

#[derive(Debug, Deserialize)]
pub struct VqvaeConfig {
    pub channels: usize,
    pub num_tokens: usize,
    pub hidden_dim: usize,
    pub num_resnet_blocks: usize,
    pub codebook_dim: usize,
    pub num_layers: usize,
    pub positional_dims: usize,
    pub kernel_size: usize,
    pub smooth_l1_loss: bool,
    pub use_transposed_convs: bool,
}

#[derive(Debug, Deserialize)]
pub struct BigvganConfig {
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub speaker_embedding_dim: usize,
    pub gpt_dim: usize,
    pub num_mels: usize,
    pub sampling_rate: usize,
    pub n_fft: usize,
    pub hop_size: usize,
    pub win_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct DatasetConfig {
    pub bpe_model: String,
    pub sample_rate: usize,
    pub mel: MelConfig,
}

#[derive(Debug, Deserialize)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub mel_fmin: usize,
    pub normalize: bool,
}

impl IndexTtsConfig {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let content = std::fs::read_to_string(&path)
            .map_err(|e| Error::Config(format!("failed to read config.json: {e}")))?;
        serde_json::from_str(&content).map_err(Error::Json)
    }
}
