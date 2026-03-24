//! CLI example: synthesize speech with IndexTTS 1.5.
//!
//! Usage:
//!   cargo run --release -p index-tts-1.5-mlx --example synthesize -- \
//!       --model ~/.OminiX/models/IndexTTS-1.5 \
//!       --reference reference.wav \
//!       --text "你好，世界！" \
//!       --output output.wav

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "synthesize", about = "IndexTTS 1.5 synthesis")]
struct Args {
    /// Path to model directory (contains model.safetensors + config.json)
    #[arg(short, long)]
    model: PathBuf,

    /// Reference audio WAV file (24kHz recommended)
    #[arg(short, long)]
    reference: PathBuf,

    /// Text to synthesize
    #[arg(short, long)]
    text: String,

    /// Output WAV file
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Top-k sampling
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Temperature
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let mut model = index_tts_15_mlx::IndexTts::load(&args.model)?;
    let ref_audio = index_tts_15_mlx::load_wav(&args.reference)?;

    let opts = index_tts_15_mlx::SynthesizeOptions {
        top_k: args.top_k,
        temperature: args.temperature,
        ..Default::default()
    };

    let samples = model.synthesize(&args.text, &ref_audio, &opts)?;
    index_tts_15_mlx::save_wav(&samples, model.sample_rate(), &args.output)?;

    println!("Saved to {:?}", args.output);
    Ok(())
}
