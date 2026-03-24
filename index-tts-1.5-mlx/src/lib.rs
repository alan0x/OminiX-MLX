//! IndexTTS 1.5 — Zero-shot voice cloning TTS on Apple Silicon using MLX.
//!
//! Model: `mlx-community/IndexTTS-1.5`
//! Architecture: Conformer Perceiver conditioner + GPT-2 decoder (24 layers) + BigVGAN vocoder
//! Sample rate: 24kHz
//!
//! # Quick start
//!
//! ```no_run
//! use index_tts_15_mlx::{IndexTts, SynthesizeOptions};
//!
//! let mut model = IndexTts::load("~/.OminiX/models/IndexTTS-1.5")?;
//! let ref_audio = index_tts_15_mlx::load_wav("reference.wav")?;
//! let opts = SynthesizeOptions::default();
//! let samples = model.synthesize("你好，世界！", &ref_audio, &opts)?;
//! index_tts_15_mlx::save_wav(&samples, model.sample_rate(), "output.wav")?;
//! # Ok::<(), index_tts_15_mlx::Error>(())
//! ```

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod bigvgan;
pub mod conditioner;
pub mod config;
pub mod error;
pub mod gpt;
pub mod mel;
pub mod sampling;

pub use error::{Error, Result};

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use tracing::info;

use bigvgan::BigVgan;
use conditioner::{ConformerEncoder, PerceiverResampler};
use config::IndexTtsConfig;
use gpt::Gpt;
use mel::{array_to_vec, mel_spectrogram};

// ─── Public API types ─────────────────────────────────────────────────────────

/// Options for synthesis.
#[derive(Debug, Clone)]
pub struct SynthesizeOptions {
    /// Top-k sampling for GPT autoregressive decoding (default: 50)
    pub top_k: usize,
    /// Top-p (nucleus) sampling (default: 0.85)
    pub top_p: f32,
    /// Temperature for sampling (default: 1.0)
    pub temperature: f32,
    /// Max mel tokens to generate (default: from config)
    pub max_mel_tokens: Option<usize>,
}

impl Default for SynthesizeOptions {
    fn default() -> Self {
        Self {
            top_k: 50,
            top_p: 0.85,
            temperature: 1.0,
            max_mel_tokens: None,
        }
    }
}

// ─── Main model struct ────────────────────────────────────────────────────────

/// High-level IndexTTS 1.5 synthesizer.
///
/// Wraps the full pipeline:
///   text → tokenizer → GPT (conditioned on reference mel) → BigVGAN → waveform
pub struct IndexTts {
    config: IndexTtsConfig,
    tokenizer: tokenizers::Tokenizer,
    conformer: ConformerEncoder,
    perceiver: PerceiverResampler,
    gpt: Gpt,
    bigvgan: BigVgan,
}

impl IndexTts {
    /// Load model from directory containing `model.safetensors` and `config.json`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        info!("Loading IndexTTS 1.5 config…");
        let config = IndexTtsConfig::load(model_dir)?;

        info!("Loading tokenizer…");
        let tokenizer = load_tokenizer(model_dir)?;

        info!("Loading model weights…");
        let weights = load_all_weights(model_dir)?;

        info!("Building conformer encoder…");
        let conformer = ConformerEncoder::load(
            &weights,
            &config.gpt.condition_module,
            config.dataset.mel.n_mels,
        )?;

        info!("Building perceiver resampler…");
        let perceiver = PerceiverResampler::load(&weights, &config.gpt.condition_module)?;

        info!("Building GPT decoder…");
        let gpt = Gpt::load(&weights, &config.gpt)?;

        info!("Building BigVGAN vocoder…");
        let bigvgan = BigVgan::load(&weights, &config.bigvgan)?;

        info!("IndexTTS 1.5 ready.");
        Ok(Self { config, tokenizer, conformer, perceiver, gpt, bigvgan })
    }

    /// Output sample rate (24000 Hz).
    pub fn sample_rate(&self) -> u32 {
        self.config.dataset.sample_rate as u32
    }

    /// Synthesize text conditioned on reference audio.
    ///
    /// `reference_audio`: 24kHz mono PCM f32 samples.
    /// Returns 24kHz mono PCM f32 samples.
    pub fn synthesize(
        &mut self,
        text: &str,
        reference_audio: &[f32],
        opts: &SynthesizeOptions,
    ) -> Result<Vec<f32>> {
        let max_mel = opts
            .max_mel_tokens
            .unwrap_or(self.config.gpt.max_mel_tokens);

        // 1. Tokenize text
        let text_ids = self.tokenize(text)?;

        // 2. Reference mel spectrogram [1, n_mels, T_ref]
        let ref_mel = mel_spectrogram(reference_audio)?;

        // Transpose to channels-last [1, T_ref, n_mels] for MLX conv
        let ref_mel_cl = ref_mel.transpose_axes(&[0, 2, 1])?;

        // 3. Conformer encoder: [1, T_ref, n_mels] → [1, T', d_enc]
        let enc = self.conformer.forward(&ref_mel)?; // conditioner takes [1, n_mels, T]

        // 4. Perceiver resampler: [1, T', d_enc] → [1, N_lat, d_model]
        let conditioning = self.perceiver.forward(&enc)?;

        // 5. GPT autoregressive generation
        let (mel_codes, gpt_latent) = self.gpt.generate(
            &text_ids,
            &conditioning,
            max_mel,
            opts.top_k,
            opts.top_p,
            opts.temperature,
        )?;

        if mel_codes.is_empty() {
            return Err(Error::Model("GPT generated zero mel tokens".into()));
        }

        // 6. BigVGAN: gpt_latent + reference_mel → waveform
        let waveform = self.bigvgan.forward(&gpt_latent, &ref_mel_cl)?;

        // 7. Flatten to Vec<f32>
        let samples = array_to_vec(&waveform)?;
        Ok(samples)
    }

    // ─── Tokenization ─────────────────────────────────────────────────────────

    /// Tokenize text, wrap with start/stop tokens, return [1, L] i32 Array.
    fn tokenize(&self, text: &str) -> Result<Array> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(format!("encode failed: {e}")))?;

        let ids = encoding.get_ids();
        let start = self.gpt.start_text_token;
        let stop = self.gpt.stop_text_token;

        // Layout: [start_text | token... | stop_text]
        let mut full: Vec<i32> = Vec::with_capacity(ids.len() + 2);
        full.push(start as i32);
        full.extend(ids.iter().map(|&id| id as i32));
        full.push(stop as i32);

        let len = full.len() as i32;
        Ok(Array::from_slice(&full, &[1, len]))
    }
}

// ─── Weight loading ───────────────────────────────────────────────────────────

/// Load all weights from either a single `model.safetensors` or a sharded set.
pub fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>> {
    // Sharded
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&json)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| Error::WeightLoad("Invalid weight index".into()))?;

        let files: std::collections::HashSet<&str> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .collect();

        let mut all: HashMap<String, Array> = HashMap::new();
        for file in files {
            let path = model_dir.join(file);
            let loaded = Array::load_safetensors(&path)
                .map_err(|e| Error::WeightLoad(format!("load {file}: {e}")))?;
            all.extend(loaded);
        }
        return Ok(all);
    }

    // Single file
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Array::load_safetensors(&single)
            .map_err(|e| Error::WeightLoad(format!("load model.safetensors: {e}")));
    }

    Err(Error::WeightLoad("No model.safetensors found in directory".into()))
}

// ─── Tokenizer loading ────────────────────────────────────────────────────────

fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Error::Tokenizer(format!("failed to load tokenizer.json: {e}")))
}

// ─── WAV I/O ──────────────────────────────────────────────────────────────────

/// Load a WAV file as mono f32 samples (resampled to 24kHz if needed).
///
/// Note: this does NOT resample — caller must provide 24kHz audio.
pub fn load_wav(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path.as_ref())
        .map_err(|e| Error::Audio(format!("failed to open wav: {e}")))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| Error::Audio(e.to_string())))
            .collect::<Result<_>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max).map_err(|e| Error::Audio(e.to_string())))
                .collect::<Result<_>>()?
        }
    };
    // If stereo, downmix to mono
    let samples = if spec.channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) * 0.5).collect()
    } else {
        samples
    };
    Ok(samples)
}

/// Save f32 samples as a mono WAV file at the given sample rate.
pub fn save_wav(samples: &[f32], sample_rate: u32, path: impl AsRef<Path>) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path.as_ref(), spec)
        .map_err(|e| Error::Audio(format!("failed to create wav: {e}")))?;
    for &s in samples {
        writer
            .write_sample(s)
            .map_err(|e| Error::Audio(format!("failed to write sample: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| Error::Audio(format!("failed to finalize wav: {e}")))?;
    Ok(())
}
