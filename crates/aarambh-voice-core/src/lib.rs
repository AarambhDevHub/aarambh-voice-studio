//! `src/lib.rs` — crate root for `aarambh-voice-core`.
//!
//! Re-exports every public type so callers can write
//! `use aarambh_voice_core::ModelConfig` instead of
//! `use aarambh_voice_core::config::ModelConfig`.
//!
//! # Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`config`] | [`AudioDomain`], [`ModelConfig`] with scale presets |
//! | [`error`]  | [`AarambhVoiceError`], one variant per crate |
//! | [`request`]| [`NaadRequest`], per-engine specs, [`AudioOutputFormat`] |

pub mod config;
pub mod error;
pub mod request;

pub use config::{AudioDomain, ModelConfig};
pub use error::AarambhVoiceError;
pub use request::{
    Alignment, AudioOutputFormat, ConsentSpec, EmotionLabel, EmotionSpec, MixSpec, MusicSpec,
    NaadRequest, SingingSpec, VoiceSpec,
};
