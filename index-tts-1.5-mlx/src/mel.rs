//! Mel spectrogram computation.
//!
//! Matches IndexTTS config:
//!   sample_rate=24000, n_fft=1024, hop=256, win=1024, n_mels=100, fmin=0

use mlx_rs::{array, ops, Array};

use crate::error::{Error, Result};

pub const SAMPLE_RATE: u32 = 24000;
pub const N_FFT: usize = 1024;
pub const HOP_LENGTH: usize = 256;
pub const WIN_LENGTH: usize = 1024;
pub const N_MELS: usize = 100;
pub const F_MIN: f32 = 0.0;
pub const F_MAX: f32 = SAMPLE_RATE as f32 / 2.0;

/// Compute mel spectrogram from raw PCM samples.
///
/// Input:  `samples` — mono f32 PCM at 24kHz
/// Output: `Array` of shape `[1, N_MELS, T]`, dtype f32
pub fn mel_spectrogram(samples: &[f32]) -> Result<Array> {
    let n = samples.len();
    if n == 0 {
        return Err(Error::Audio("empty audio input".into()));
    }

    // Pad signal: center-pad by n_fft/2 on each side (matches librosa default)
    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; n + 2 * pad];
    padded[pad..pad + n].copy_from_slice(samples);

    let audio = Array::from_slice(&padded, &[1, padded.len() as i32]);

    // Build mel filterbank [n_mels, n_fft/2 + 1]
    let filters = mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE as f32, F_MIN, F_MAX);
    let filters = Array::from_slice(&filters, &[N_MELS as i32, (N_FFT / 2 + 1) as i32]);

    // STFT → power spectrum
    let stft = stft_power(&audio, N_FFT, HOP_LENGTH, WIN_LENGTH)?;
    // stft: [1, n_frames, n_fft/2+1]

    // Apply mel filterbank: [1, n_frames, n_mels]
    let mel = ops::matmul(&stft, &filters.t())?;

    // Clamp to avoid log(0), then log
    let mel = ops::maximum(&mel, &array!(1e-5f32))?;
    let mel = ops::log(&mel)?;

    // Transpose to [1, n_mels, n_frames]
    let mel = mel.transpose_axes(&[0, 2, 1])?;
    ops::eval(&mel)?;

    Ok(mel)
}

// ─── STFT ────────────────────────────────────────────────────────────────────

/// Compute power spectrogram via STFT.
///
/// Returns `[1, n_frames, n_fft/2+1]` (power = |X|²).
fn stft_power(audio: &Array, n_fft: usize, hop: usize, win: usize) -> Result<Array> {
    // Hann window
    let window = hann_window(win);

    let audio_len = audio.shape()[1] as usize;
    let n_frames = (audio_len.saturating_sub(n_fft)) / hop + 1;
    let n_bins = n_fft / 2 + 1;

    // Frame the signal: read audio data back to CPU, then window+frame it.
    let mut frames = vec![0.0f32; n_frames * win];
    let audio_flat = array_to_vec(audio)?;

    for f in 0..n_frames {
        let start = f * hop;
        for i in 0..win {
            let src = start + i;
            frames[f * win + i] = if src < audio_flat.len() {
                audio_flat[src] * window[i]
            } else {
                0.0
            };
        }
    }

    // FFT each frame using MLX rfft
    let frames_arr = Array::from_slice(&frames, &[1, n_frames as i32, win as i32]);
    // MLX rfft: input [..., N] → output [..., N/2+1] complex64
    let spectrum = mlx_rs::fft::rfft(&frames_arr, None, -1)?;
    // spectrum: [1, n_frames, n_fft/2+1] complex

    // Power spectrum: |X|² via magnitude squared
    let mag = ops::abs(&spectrum)?;
    let power = (&mag * &mag)?;
    // [1, n_frames, n_bins]

    Ok(power)
}

// ─── Hann window ─────────────────────────────────────────────────────────────

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())
        })
        .collect()
}

// ─── Mel filterbank ──────────────────────────────────────────────────────────

/// Build mel filterbank matrix `[n_mels, n_fft/2+1]` (HTK formula).
fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f32, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let fft_freqs: Vec<f32> = (0..n_bins)
        .map(|i| i as f32 * sr / n_fft as f32)
        .collect();

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];
        for (k, &fk) in fft_freqs.iter().enumerate() {
            let val = if fk >= f_left && fk <= f_center {
                (fk - f_left) / (f_center - f_left)
            } else if fk > f_center && fk <= f_right {
                (f_right - fk) / (f_right - f_center)
            } else {
                0.0
            };
            filters[m * n_bins + k] = val;
        }
    }
    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

// ─── Helper ──────────────────────────────────────────────────────────────────

/// Read an MLX Array into a Rust Vec<f32>.
pub fn array_to_vec(arr: &Array) -> Result<Vec<f32>> {
    ops::eval(arr)?;
    Ok(arr.as_slice::<f32>()
        .map_err(|e| Error::Model(format!("array_to_vec failed: {e}")))?
        .to_vec())
}
