//! Token sampling for GPT autoregressive decoding.
//!
//! Supports temperature scaling, top-k, and top-p (nucleus) filtering.
//! Uses GPU-resident MLX operations to avoid unnecessary CPU roundtrips.

use mlx_rs::{array, ops, Array};

use crate::error::{Error, Result};

/// Sample the next token from logits using temperature + top-k + top-p.
///
/// `logits`: `[vocab_size]` raw (un-softmaxed) scores.
/// Returns a single token id `u32`.
pub fn sample(
    logits: &Array,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<u32> {
    // Greedy decode when temperature is effectively zero
    if temperature < 1e-6 {
        let idx = ops::indexing::argmax_axis(logits, -1, None)?;
        ops::eval(&idx)?;
        return Ok(idx.item::<u32>());
    }

    // Temperature scaling
    let mut logits = logits.multiply(array!(1.0f32 / temperature))?;

    // Top-k filtering (GPU-resident)
    if top_k > 0 {
        logits = apply_top_k(&logits, top_k as i32)?;
    }

    // Top-p (nucleus) filtering (GPU-resident)
    if top_p < 1.0 {
        logits = apply_top_p(&logits, top_p)?;
    }

    // Categorical sampling
    let token = mlx_rs::random::categorical(&logits, None, None, None)?;
    ops::eval(&token)?;
    Ok(token.item::<u32>())
}

/// Keep only the top-k largest logits; mask the rest to -inf.
fn apply_top_k(logits: &Array, k: i32) -> Result<Array> {
    let top_values = ops::indexing::topk(logits, k)?;
    let threshold = top_values.min_axis(-1, true)?;
    let below = logits.lt(&threshold)?;
    ops::r#where(&below, &array!(f32::NEG_INFINITY), logits)
        .map_err(Error::Mlx)
}

/// Keep the smallest set of tokens whose cumulative probability >= p; mask rest to -inf.
fn apply_top_p(logits: &Array, p: f32) -> Result<Array> {
    // Sort descending: negate → sort ascending → negate back
    let sorted = ops::sort_axis(&logits.negative()?, -1)?.negative()?;

    // Softmax on sorted logits → probabilities in descending order
    let sorted_probs = ops::softmax_axis(&sorted, -1, None::<bool>)?;

    // Cumulative sum
    let cum_probs = ops::cumsum(&sorted_probs, Some(-1i32), None::<bool>, None::<bool>)?;

    // Remove tokens where cumsum-before-this-token >= p
    let shifted = cum_probs.subtract(&sorted_probs)?;
    let to_remove = shifted.ge(&array!(p))?;

    // Minimum kept logit value in sorted space → threshold
    let kept = ops::r#where(&to_remove, &array!(f32::MAX), &sorted)?;
    let threshold = kept.min_axis(-1, true)?;

    // Apply to original (unsorted) logits
    let below = logits.lt(&threshold)?;
    ops::r#where(&below, &array!(f32::NEG_INFINITY), logits)
        .map_err(Error::Mlx)
}
