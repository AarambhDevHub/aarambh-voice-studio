//! `src/error.rs` — centralised error type for aarambh-voice-studio.
//!
//! Defines [`AarambhVoiceError`], the single error enum whose variants
//! correspond to every crate in the workspace, and a convenience alias
//! [`VoiceResult<T>`] that pins the error variant.
//!
//! # Variants
//!
//! | Variant | Source | When raised |
//! |---------|--------|-------------|
//! | `Codec` | manual | encoder / decoder / RVQ forward or backward failure |
//! | `TextPrep` | manual | G2P lookup failure, normalisation error |
//! | `Data` | manual | dataset loading, preprocessing, auto-labelling failure |
//! | `Nn` | manual | transformer block, conditioning injection, activation failure |
//! | `Kernel` | manual | CPU SIMD or fused STFT kernel failure |
//! | `Model` | manual | per-engine forward pass or diffusion-head failure |
//! | `Weights` | manual | SafeTensors save / load failure |
//! | `Train` | manual | training loop (optimiser, checkpointing) failure |
//! | `Quant` | manual | INT4 / INT8 quantisation or GGUF export failure |
//! | `Finetune` | manual | LoRA / QLoRA / DoRA adapter injection or training failure |
//! | `Align` | manual | GRPO / DPO alignment loop failure |
//! | `SelfLearn` | manual | online self-learning or anti-forgetting failure |
//! | `Speaker` | manual | speaker-encoder forward pass or embedding extraction failure |
//! | `Emotion` | manual | emotion-encoder forward pass or embedding extraction failure |
//! | `Music` | manual | music understanding or generation failure |
//! | `Sing` | manual | singing synthesis or diffusion refinement failure |
//! | `Mix` | manual | mixing / mastering DSP failure |
//! | `Compose` | manual | structure planner or audio orchestration failure |
//! | `Safety` | manual | consent gating, watermarking, or guardrail failure |
//! | `Eval` | manual | evaluation harness or baseline comparison failure |
//! | `Control` | manual | NaadRequest parsing or validation failure |
//! | `Inference` | manual | KV cache, streaming, or speculative decoding failure |
//! | `Serve` | manual | HTTP server startup or request-handling failure |
//! | `Serialisation` | manual | JSON / TOML / message-pack serialisation failure |
//! | `Io` | `#[from]` | filesystem read / write failure |

use thiserror::Error;

// ---------------------------------------------------------------------------
// AarambhVoiceError
// ---------------------------------------------------------------------------

/// The single error type used across every crate in the workspace.
///
/// Each variant corresponds to one crate's failure mode. Variants that
/// wrap a bare [`String`] are constructed manually; the [`Io`] variant
/// is converted automatically via `#[from]` so callers can use the `?`
/// operator on [`std::io::Error`].
///
/// [`Io`]: Self::Io
#[derive(Debug, Error)]
pub enum AarambhVoiceError {
    /// `aarambh-voice-codec` — encoder, decoder, or RVQ forward/backward failure.
    #[error("Codec error: {0}")]
    Codec(String),

    /// `aarambh-voice-textprep` — G2P lookup or text-normalisation failure.
    #[error("Text prep error: {0}")]
    TextPrep(String),

    /// `aarambh-voice-data` — dataset loading, preprocessing, or auto-labelling failure.
    #[error("Data pipeline error: {0}")]
    Data(String),

    /// `aarambh-voice-nn` — transformer block, conditioning injection, or activation failure.
    #[error("Neural network error: {0}")]
    Nn(String),

    /// `aarambh-voice-kernel` — CPU SIMD kernel or fused STFT failure.
    #[error("Kernel error: {0}")]
    Kernel(String),

    /// `aarambh-voice-model` — per-engine forward pass or diffusion-head failure.
    #[error("Model error: {0}")]
    Model(String),

    /// `aarambh-voice-weights` — SafeTensors save or load failure.
    #[error("Weights error: {0}")]
    Weights(String),

    /// `aarambh-voice-train` — training-loop failure (optimiser, checkpointing).
    #[error("Training error: {0}")]
    Train(String),

    /// `aarambh-voice-quant` — INT4/INT8 quantisation or GGUF export failure.
    #[error("Quantisation error: {0}")]
    Quant(String),

    /// `aarambh-voice-finetune` — LoRA/QLoRA/DoRA adapter injection or training failure.
    #[error("Fine-tuning error: {0}")]
    Finetune(String),

    /// `aarambh-voice-align` — GRPO or DPO alignment loop failure.
    #[error("Alignment error: {0}")]
    Align(String),

    /// `aarambh-voice-selflearn` — online self-learning or anti-forgetting failure.
    #[error("Self-learning error: {0}")]
    SelfLearn(String),

    /// `aarambh-voice-speaker` — speaker-encoder forward pass or embedding extraction failure.
    #[error("Speaker error: {0}")]
    Speaker(String),

    /// `aarambh-voice-emotion` — emotion-encoder forward pass or embedding extraction failure.
    #[error("Emotion error: {0}")]
    Emotion(String),

    /// `aarambh-voice-music` — music-understanding classifier or music-generation failure.
    #[error("Music error: {0}")]
    Music(String),

    /// `aarambh-voice-sing` — singing synthesis or diffusion-refinement failure.
    #[error("Singing error: {0}")]
    Sing(String),

    /// `aarambh-voice-mix` — mixing or mastering DSP failure.
    #[error("Mixing error: {0}")]
    Mix(String),

    /// `aarambh-voice-compose` — structure planner or audio orchestration failure.
    #[error("Composition error: {0}")]
    Compose(String),

    /// `aarambh-voice-safety` — consent gating, watermarking, or guardrail failure.
    #[error("Safety error: {0}")]
    Safety(String),

    /// `aarambh-voice-eval` — evaluation harness or baseline-comparison failure.
    #[error("Evaluation error: {0}")]
    Eval(String),

    /// `aarambh-voice-control` — NaadRequest parsing or validation failure.
    #[error("Control layer error: {0}")]
    Control(String),

    /// `aarambh-voice-inference` — KV cache, streaming, or speculative-decoding failure.
    #[error("Inference error: {0}")]
    Inference(String),

    /// `aarambh-voice-serve` — HTTP server startup or request-handling failure.
    #[error("Server error: {0}")]
    Serve(String),

    /// A JSON / TOML / message-pack serialisation or deserialisation error.
    #[error("Serialisation error: {0}")]
    Serialisation(String),

    /// A filesystem read or write failed.
    ///
    /// Automatically converted from [`std::io::Error`] via `#[from]`, so
    /// any `std::fs` or `std::io` operation inside a `Result`-returning
    /// function can use `?` directly.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Result alias
// ---------------------------------------------------------------------------

/// Crate-wide `Result` alias — pins the `Err` variant to [`AarambhVoiceError`].
///
/// Import with `use crate::error::VoiceResult;` in any module that returns
/// fallible values.
pub type VoiceResult<T> = std::result::Result<T, AarambhVoiceError>;
