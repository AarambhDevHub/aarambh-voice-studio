//! `src/request.rs` — the full control-surface types for aarambh-voice-studio.
//!
//! Every parameter the system accepts is an explicit typed field on
//! [`NaadRequest`] — no hidden presets, no magic strings. The per-engine
//! substructs (`VoiceSpec`, `EmotionSpec`, …) are stubs here and will be
//! fleshed out with full fields in Phase 18 (ARCHITECTURE.md §16).
//!
//! # Request flow
//!
//! ```text
//! NaadRequest
//!   ├── text          (the words to speak / sing)
//!   ├── voice         (reference clip, design text, or speaker ID)
//!   ├── emotion       (label, free-text description, intensity)
//!   ├── singing?      (melody + lyrics alignment)
//!   ├── music?        (style prompt + duration)
//!   ├── mix?          (LUFS target + vocal gain)
//!   ├── output_format (WAV / FLAC / Opus / MP3)
//!   └── consent       (required when voice.reference_audio is set)
//! ```

// ---------------------------------------------------------------------------
// VoiceSpec
// ---------------------------------------------------------------------------

/// Specification of which voice to use for speech generation.
///
/// Exactly one of the three fields should be `Some`:
/// - [`reference_audio`](Self::reference_audio) for zero-shot cloning,
/// - [`design_text`](Self::design_text) for text-described voice design,
/// - [`speaker_id`](Self::speaker_id) for a previously saved speaker embedding.
///
/// If all three are `None` the engine falls back to a default voice.
#[derive(Debug, Clone)]
pub struct VoiceSpec {
    /// Path to a short reference audio clip (3–10 s) whose voice will be cloned.
    ///
    /// When set, [`ConsentSpec::consent_provided`] must be `true`.
    pub reference_audio: Option<String>,

    /// A free-text description of the desired voice, e.g. "warm, gravelly,
    /// middle-aged male".
    pub design_text: Option<String>,

    /// A previously saved speaker-embedding key (returned by the speaker
    /// encoder during a prior cloning or design session).
    pub speaker_id: Option<String>,
}

// ---------------------------------------------------------------------------
// EmotionLabel & EmotionSpec
// ---------------------------------------------------------------------------

/// Discrete emotion categories recognised by the emotion encoder.
///
/// Used when [`EmotionSpec::label`] is provided instead of (or alongside)
/// a free-text [`EmotionSpec::description`]. The eight categories cover
/// the basic emotions used in RAVDESS / CREMA-D style corpora.
#[derive(Debug, Clone)]
pub enum EmotionLabel {
    /// Joyful, cheerful, positive affect.
    Happy,
    /// Downcast, sorrowful, negative affect.
    Sad,
    /// Hostile, frustrated, agitated.
    Angry,
    /// Baseline — no strong emotional valence.
    Neutral,
    /// Startled, amazed, caught off-guard.
    Surprised,
    /// Anxious, scared, threatened.
    Fearful,
    /// Repulsed, grossed-out, strong aversion.
    Disgusted,
    /// Relaxed, peaceful, low-arousal positive affect.
    Calm,
}

/// Continuous emotion-control parameters.
///
/// Emotion can be specified as a discrete label, a free-text description
/// (e.g. "nervous but trying to hide it"), or both. The intensity scalar
/// scales the embedding magnitude regardless of which source is used.
#[derive(Debug, Clone)]
pub struct EmotionSpec {
    /// A discrete emotion category from [`EmotionLabel`].
    ///
    /// Mapped into the 8-dim emotion embedding space via a learned lookup.
    pub label: Option<EmotionLabel>,

    /// A free-text description of the desired emotion.
    ///
    /// Encoded by the emotion encoder's text-projection head (same small
    /// MLP that maps discrete labels into the continuous space).
    pub description: Option<String>,

    /// Emotion intensity, from 0.0 (neutral / flat) to 1.0 (maximum
    /// expression for the given label or description).
    pub intensity: f32,
}

// ---------------------------------------------------------------------------
// SingingSpec
// ---------------------------------------------------------------------------

/// Per-frame alignment of a lyric syllable to a melody note.
///
/// Produced by the forced-alignment step inside `aarambh-voice-textprep`
/// and consumed by the singing engine's duration/pitch heads.
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Codec-frame index where this syllable starts.
    pub start_frame: usize,
    /// Codec-frame index where this syllable ends (exclusive).
    pub end_frame: usize,
}

/// Parameters for the singing engine.
///
/// When `singing` is `Some` on [`NaadRequest`], the system generates sung
/// vocals instead of spoken speech. The melody is provided as raw MIDI-like
/// byte data; alignment to the lyrics is optional — if omitted the engine
/// aligns automatically.
#[derive(Debug, Clone)]
pub struct SingingSpec {
    /// MIDI-like melody data (raw bytes, format TBD in Phase 13).
    ///
    /// Contains note-on/note-off events with pitch and timing.
    pub melody: Vec<u8>,

    /// Optional per-syllable alignment to the lyrics text.
    ///
    /// If `None`, the singing engine runs its own lightweight forced-alignment
    /// pass to determine timing.
    pub lyrics_alignment: Option<Vec<Alignment>>,
}

// ---------------------------------------------------------------------------
// MusicSpec
// ---------------------------------------------------------------------------

/// Parameters for the background-music generation engine.
///
/// When `music` is `Some` on [`NaadRequest`], instrumental music is
/// generated alongside (or instead of) voice content. The style prompt
/// uses the same tagging vocabulary as the music-understanding classifier
/// from Phase 11.
#[derive(Debug, Clone)]
pub struct MusicSpec {
    /// Free-text style description, e.g. "lo-fi hip-hop, rainy, 80 bpm".
    ///
    /// Encoded by the same tag vocabulary used by the music-understanding
    /// classifier so that auto-labelled training data and inference-time
    /// prompts share a common embedding space.
    pub style_prompt: String,

    /// Target duration of the generated music in seconds.
    ///
    /// The generation engine produces roughly this many seconds of audio;
    /// exact length may vary by a fraction of a frame at 12.5 Hz.
    pub duration_seconds: f32,
}

// ---------------------------------------------------------------------------
// MixSpec
// ---------------------------------------------------------------------------

/// Parameters for the mixing / mastering stage.
///
/// When both voice/singing and music are present, this spec controls how
/// the two streams are blended and normalised. If `mix` is `None`, a
/// default mix (0 dB vocal gain, −14 LUFS integrated target) is applied.
#[derive(Debug, Clone)]
pub struct MixSpec {
    /// Integrated LUFS target for the final mix.
    ///
    /// Standard values: −14 LUFS (streaming normalisation), −16 LUFS
    /// (broadcast), −23 LUFS (EBU R128). Range approximately −30 to −10.
    pub target_lufs: f32,

    /// Vocal gain relative to the instrumental backing, in dB.
    ///
    /// Positive values push the vocal forward; negative values blend it
    /// further into the mix. Typical range: −6 dB to +6 dB.
    pub vocal_gain_db: f32,
}

// ---------------------------------------------------------------------------
// AudioOutputFormat
// ---------------------------------------------------------------------------

/// The container / codec format for generated audio output.
///
/// All formats are wrapped in a RIFF / Ogg container as appropriate;
/// the exact encoder implementation lives in `aarambh-voice-codec` and
/// uses the DSP libraries specified in ARCHITECTURE.md §3.
///
/// # Feature gates
///
/// | Variant | Library | Default? | Notes |
/// |---------|---------|----------|-------|
/// | `Wav`  | `hound` | yes | Uncompressed PCM, always available |
/// | `Flac` | `flacenc` | yes | Lossless compression |
/// | `Opus` | `audiopus` | yes | Low-latency lossy codec |
/// | `Mp3`  | `mp3lame-encoder` | behind `mp3` feature | LGPL / patent-encumbered |
#[derive(Debug, Clone)]
pub enum AudioOutputFormat {
    /// Uncompressed 16-bit / 24-bit PCM in a WAV container.
    Wav,
    /// FLAC lossless compression (pure Rust, always available).
    Flac,
    /// Opus low-latency lossy compression via libopus bindings.
    Opus,
    /// MP3 lossy compression via LAME (feature-gated, see licensing notes).
    Mp3,
}

// ---------------------------------------------------------------------------
// ConsentSpec
// ---------------------------------------------------------------------------

/// Consent attestation for voice cloning.
///
/// Required whenever [`VoiceSpec::reference_audio`] is set — the system
/// refuses to clone a voice without explicit consent. The crate-level
/// safety layer (`aarambh-voice-safety`) enforces this at runtime.
///
/// # Future
///
/// Phase 19 will extend this with a watermarking payload that ties the
/// generated clip back to the consent record.
#[derive(Debug, Clone)]
pub struct ConsentSpec {
    /// Whether consent has been explicitly provided.
    ///
    /// When `false`, any request with `voice.reference_audio` set is
    /// rejected before any model inference runs.
    pub consent_provided: bool,

    /// An identifier linking this generation back to a stored consent record.
    ///
    /// Optional at the request level; the safety layer may require it
    /// depending on the system's consent-policy configuration.
    pub speaker_id: Option<String>,
}

// ---------------------------------------------------------------------------
// NaadRequest
// ---------------------------------------------------------------------------

/// The single top-level request type for every engine in the system.
///
/// Every generation begins with a [`NaadRequest`] — there is no second
/// entry path. The request specifies exactly what to generate and how to
/// format the result. Fields that are `None` are simply not activated.
///
/// # Examples
///
/// **Plain TTS (default voice, no emotion):**
/// ```rust,ignore
/// NaadRequest {
///     text: "Hello world".into(),
///     voice: VoiceSpec { reference_audio: None, design_text: None, speaker_id: None },
///     emotion: EmotionSpec { label: None, description: None, intensity: 0.0 },
///     singing: None,
///     music: None,
///     mix: None,
///     output_format: AudioOutputFormat::Wav,
///     consent: ConsentSpec { consent_provided: false, speaker_id: None },
/// }
/// ```
///
/// **Voice cloning with background music:**
/// ```rust,ignore
/// NaadRequest {
///     text: "Hello world".into(),
///     voice: VoiceSpec { reference_audio: Some("speaker.wav".into()), .. },
///     music: Some(MusicSpec { style_prompt: "calm piano".into(), duration_seconds: 10.0 }),
///     .. // emotion, mix, etc. as needed
/// }
/// ```
#[derive(Debug, Clone)]
pub struct NaadRequest {
    /// The text content to be spoken or sung.
    pub text: String,

    /// Voice specification (cloning, design, or saved ID).
    pub voice: VoiceSpec,

    /// Continuous emotion-control parameters.
    pub emotion: EmotionSpec,

    /// Singing parameters — `None` means speech mode.
    pub singing: Option<SingingSpec>,

    /// Background music parameters — `None` means no instrumental backing.
    pub music: Option<MusicSpec>,

    /// Mixing / mastering parameters — `None` means default mix.
    pub mix: Option<MixSpec>,

    /// Output audio container / codec format.
    pub output_format: AudioOutputFormat,

    /// Consent attestation (required for voice cloning).
    pub consent: ConsentSpec,
}
