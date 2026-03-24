//! IndexTTS 1.5 — Zero-shot voice cloning TTS on Apple Silicon using MLX.
//!
//! Model: `mlx-community/IndexTTS-1.5`
//! Architecture: VQVAE + GPT (24-layer) + BigVGAN vocoder
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

pub mod config;
pub mod error;

// TODO: implement in Phase 2
// pub mod mel;
// pub mod vqvae;
// pub mod gpt;
// pub mod bigvgan;
// pub mod infer;

pub use error::{Error, Result};

use std::path::Path;

/// High-level IndexTTS 1.5 synthesizer.
///
/// Wraps VQVAE + GPT + BigVGAN into a single inference interface.
pub struct IndexTts {
    // TODO: add fields in Phase 2
    _config: config::IndexTtsConfig,
}

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

impl IndexTts {
    /// Load model from directory containing `model.safetensors` and `config.json`.
    pub fn load(_model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = _model_dir.as_ref();
        let config = config::IndexTtsConfig::load(model_dir)?;
        // TODO: load weights and build sub-models in Phase 2
        Ok(Self { _config: config })
    }

    /// Output sample rate (24000 Hz).
    pub fn sample_rate(&self) -> u32 {
        self._config.dataset.sample_rate as u32
    }

    /// Synthesize text conditioned on reference audio.
    ///
    /// `reference_audio`: 24kHz mono PCM f32 samples.
    /// Returns 24kHz mono PCM f32 samples.
    pub fn synthesize(
        &mut self,
        _text: &str,
        _reference_audio: &[f32],
        _opts: &SynthesizeOptions,
    ) -> Result<Vec<f32>> {
        // TODO: implement in Phase 2
        Err(Error::Model("synthesize not yet implemented".into()))
    }
}

/// Load a WAV file as 24kHz mono f32 samples.
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
    Ok(samples)
}

/// Save f32 samples as a 24kHz mono WAV file.
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
