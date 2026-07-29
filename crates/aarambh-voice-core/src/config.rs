//! `src/config.rs` — shared configuration types for aarambh-voice-studio.
//!
//! Provides two core types used by every crate in the workspace:
//!
//! - [`AudioDomain`] — discriminates Speech / Music / Singing at the type level.
//! - [`ModelConfig`] — transformer hyper-parameters, parameterised by
//!   [`AudioDomain`] with four preset scales.
//!
//! # Model scales (ARCHITECTURE_VOICE_STUDIO_PART1.md §5)
//!
//! | Scale | `d_model` | `n_layers` | `n_heads` | `n_kv_heads` | Params |
//! |-------|-----------|------------|-----------|--------------|--------|
//! | Tiny   | 256  | 6  | 8  | 2 | ≈ 10 M  |
//! | Small  | 512  | 12 | 8  | 4 | ≈ 55 M  |
//! | Medium | 768  | 18 | 12 | 4 | ≈ 170 M |
//! | Large  | 1024 | 24 | 16 | 4 | ≈ 450 M |
//!
//! All scales share `n_codebooks = 8` and `max_frames` scaled proportionally
//! to the architecture's context window.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// AudioDomain
// ---------------------------------------------------------------------------

/// The three audio domains the system can process or generate.
///
/// Every [`ModelConfig`] is parameterised by a domain so that the same
/// architecture can be instantiated for any of the three tasks without
/// duplicating config structs.
///
/// # Serialisation
///
/// Serialised as a plain string (`"Speech"`, `"Music"`, `"Singing"`) via
/// `serde` derives — no special tag handling needed for this level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioDomain {
    /// Spoken-language audio: TTS, voice cloning, emotion control.
    Speech,
    /// Instrumental music: understanding (classifier) and generation.
    Music,
    /// Sung vocals with melody conditioning (lyrics + MIDI-like input).
    Singing,
}

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

/// Hyper-parameters that fully describe a transformer model for voice-studio.
///
/// Every engine (TTS, music, singing) shares this same struct, differing only
/// in [`AudioDomain`] and the path through which conditioning signals are
/// injected. The struct is serialisable so checkpoints carry their own config.
///
/// # Constraint
///
/// `d_model` must be divisible by `n_heads`. The training pipeline validates
/// this before constructing any tensors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Embedding dimension (width of every hidden state and attention output).
    ///
    /// Must be divisible by [`n_heads`](Self::n_heads).
    /// Tiny = 256, Small = 512, Medium = 768, Large = 1024.
    pub d_model: usize,

    /// Number of transformer blocks stacked in the decoder.
    ///
    /// Each block contains one Grouped-Query Attention layer and one SwiGLU
    /// feed-forward network.
    /// Tiny = 6, Small = 12, Medium = 18, Large = 24.
    pub n_layers: usize,

    /// Number of attention heads per block.
    ///
    /// Head dimension = `d_model / n_heads`.
    /// Tiny = 8 (head_dim 32), Small = 8 (head_dim 64),
    /// Medium = 12 (head_dim 64), Large = 16 (head_dim 64).
    pub n_heads: usize,

    /// Number of key/value heads for Grouped-Query Attention (GQA).
    ///
    /// When `n_kv_heads < n_heads`, every `n_heads / n_kv_heads` query heads
    /// share one key/value pair, saving memory with a small quality trade-off.
    /// Tiny = 2, Small = 4, Medium = 4, Large = 4.
    pub n_kv_heads: usize,

    /// Maximum number of codec-frame positions the model can attend over.
    ///
    /// At 12.5 Hz frame rate this corresponds to:
    /// - 375 frames → 30 seconds (Tiny)
    /// - 750 frames → 60 seconds (Small)
    /// - 1500 frames → 2 minutes (Medium)
    /// - 3000 frames → 4 minutes (Large)
    pub max_frames: usize,

    /// Number of Residual Vector Quantisation (RVQ) codebooks used by the codec.
    ///
    /// Each codebook adds one discrete token per frame; the language model
    /// predicts all `n_codebooks` tokens per timestep. Eight codebooks is the
    /// standard value used by modern audio codecs (EnCodec, DAC, Mimi).
    pub n_codebooks: usize,

    /// The audio domain this config targets.
    ///
    /// Determines which conditioning signals are injected and what loss
    /// weighting schedule is applied during training.
    pub domain: AudioDomain,
}

impl ModelConfig {
    // -----------------------------------------------------------------------
    // Presets
    // -----------------------------------------------------------------------

    /// Returns the **Tiny** preset: 256-dim, 6 layers, ≈ 10 M params.
    ///
    /// Designed for CPU-fast iteration on an i3 with 8 GB RAM. Use this
    /// scale for all initial experiments and for the speculative-decoding
    /// draft model.
    ///
    /// | Field | Value |
    /// |-------|-------|
    /// | `d_model` | 256 |
    /// | `n_layers` | 6 |
    /// | `n_heads` | 8 |
    /// | `n_kv_heads` | 2 |
    /// | `max_frames` | 375 (≈ 30 s at 12.5 Hz) |
    /// | `n_codebooks` | 8 |
    pub fn tiny(domain: AudioDomain) -> Self {
        Self {
            d_model: 256,
            n_layers: 6,
            n_heads: 8,
            n_kv_heads: 2,
            max_frames: 375,
            n_codebooks: 8,
            domain,
        }
    }

    /// Returns the **Small** preset: 512-dim, 12 layers, ≈ 55 M params.
    ///
    /// The default scale for end-to-end engine training when a GPU (Kaggle
    /// T4 or similar) is available. Fits comfortably in 16 GB VRAM.
    ///
    /// | Field | Value |
    /// |-------|-------|
    /// | `d_model` | 512 |
    /// | `n_layers` | 12 |
    /// | `n_heads` | 8 |
    /// | `n_kv_heads` | 4 |
    /// | `max_frames` | 750 (≈ 60 s at 12.5 Hz) |
    /// | `n_codebooks` | 8 |
    pub fn small(domain: AudioDomain) -> Self {
        Self {
            d_model: 512,
            n_layers: 12,
            n_heads: 8,
            n_kv_heads: 4,
            max_frames: 750,
            n_codebooks: 8,
            domain,
        }
    }

    /// Returns the **Medium** preset: 768-dim, 18 layers, ≈ 170 M params.
    ///
    /// Requires a GPU with ≥ 24 GB VRAM for training. The largest scale
    /// that remains practical for a solo developer on a single GPU.
    ///
    /// | Field | Value |
    /// |-------|-------|
    /// | `d_model` | 768 |
    /// | `n_layers` | 18 |
    /// | `n_heads` | 12 |
    /// | `n_kv_heads` | 4 |
    /// | `max_frames` | 1500 (≈ 2 min at 12.5 Hz) |
    /// | `n_codebooks` | 8 |
    pub fn medium(domain: AudioDomain) -> Self {
        Self {
            d_model: 768,
            n_layers: 18,
            n_heads: 12,
            n_kv_heads: 4,
            max_frames: 1500,
            n_codebooks: 8,
            domain,
        }
    }

    /// Returns the **Large** preset: 1024-dim, 24 layers, ≈ 450 M params.
    ///
    /// The v1 maximum scale. Requires multi-GPU training or a high-VRAM
    /// cloud instance (A100 80 GB). Use for the final production checkpoint
    /// after all capabilities are validated at Small or Medium.
    ///
    /// | Field | Value |
    /// |-------|-------|
    /// | `d_model` | 1024 |
    /// | `n_layers` | 24 |
    /// | `n_heads` | 16 |
    /// | `n_kv_heads` | 4 |
    /// | `max_frames` | 3000 (≈ 4 min at 12.5 Hz) |
    /// | `n_codebooks` | 8 |
    pub fn large(domain: AudioDomain) -> Self {
        Self {
            d_model: 1024,
            n_layers: 24,
            n_heads: 16,
            n_kv_heads: 4,
            max_frames: 3000,
            n_codebooks: 8,
            domain,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `ModelConfig::tiny(AudioDomain::Speech)` produces the exact
    /// field values from ARCHITECTURE.md §5.
    #[test]
    fn test_tiny_speech() {
        let cfg = ModelConfig::tiny(AudioDomain::Speech);
        assert_eq!(cfg.d_model, 256);
        assert_eq!(cfg.n_layers, 6);
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.max_frames, 375);
        assert_eq!(cfg.n_codebooks, 8);
        assert_eq!(cfg.domain, AudioDomain::Speech);
    }

    /// Verify that the Tiny preset works for all three domains.
    #[test]
    fn test_tiny_music() {
        let cfg = ModelConfig::tiny(AudioDomain::Music);
        assert_eq!(cfg.n_layers, 6);
        assert_eq!(cfg.domain, AudioDomain::Music);
    }

    /// Verify that the Tiny preset works for the Singing domain.
    #[test]
    fn test_tiny_singing() {
        let cfg = ModelConfig::tiny(AudioDomain::Singing);
        assert_eq!(cfg.n_layers, 6);
        assert_eq!(cfg.domain, AudioDomain::Singing);
    }

    /// Verify that `ModelConfig::small()` produces the expected field values.
    #[test]
    fn test_small_speech() {
        let cfg = ModelConfig::small(AudioDomain::Speech);
        assert_eq!(cfg.d_model, 512);
        assert_eq!(cfg.n_layers, 12);
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.n_kv_heads, 4);
        assert_eq!(cfg.max_frames, 750);
    }

    /// Verify that `ModelConfig::medium()` produces the expected field values.
    #[test]
    fn test_medium_speech() {
        let cfg = ModelConfig::medium(AudioDomain::Speech);
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.n_layers, 18);
        assert_eq!(cfg.n_heads, 12);
    }

    /// Verify that `ModelConfig::large()` produces the expected field values.
    #[test]
    fn test_large_speech() {
        let cfg = ModelConfig::large(AudioDomain::Speech);
        assert_eq!(cfg.d_model, 1024);
        assert_eq!(cfg.n_layers, 24);
        assert_eq!(cfg.n_heads, 16);
        assert_eq!(cfg.n_kv_heads, 4);
        assert_eq!(cfg.max_frames, 3000);
    }

    /// Round-trip through `serde_json`: serialise then deserialise must
    /// produce an identical [`ModelConfig`].
    #[test]
    fn test_config_round_trip() {
        let cfg = ModelConfig::medium(AudioDomain::Music);
        let json = serde_json::to_string(&cfg).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, deserialized);
    }

    /// Same round-trip test for the Tiny / Speech combination to catch
    /// domain-specific serialisation issues.
    #[test]
    fn test_config_round_trip_speech() {
        let cfg = ModelConfig::tiny(AudioDomain::Speech);
        let json = serde_json::to_string(&cfg).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, deserialized);
        assert_eq!(deserialized.domain, AudioDomain::Speech);
    }
}
