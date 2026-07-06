# ARCHITECTURE_VOICE_STUDIO.md — aarambh-voice-studio

> A modern, from-scratch speech, music, and singing engine in Rust using
> `candle`. Unified TTS, zero-shot voice cloning, text-described voice design,
> continuous emotion control, background music generation and understanding,
> a cappella and accompanied singing synthesis, and a lyrics-to-song composer
> — all sitting on a shared neural-audio-codec transformer core, exposed
> through one full control API. Sibling project to `aarambh-ai`, reusing its
> transformer primitives and training/fine-tuning discipline wherever
> possible instead of re-inventing them.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Dependency Versions & Toolchain](#3-dependency-versions--toolchain)
4. [Complete Workspace — 20 Library Crates](#4-complete-workspace--20-library-crates)
5. [Model Scales](#5-model-scales)
6. [The Full Journey: Text/Audio → Output](#6-the-full-journey-textaudio--output)
   - 6.1 Neural Audio Codec (Tokenising Sound)
   - 6.2 Shared Transformer Core
   - 6.3 Speaker Conditioning — Zero-Shot Voice Cloning
   - 6.4 Voice Design — Text-Described Voice
   - 6.5 Emotion Conditioning
   - 6.6 Music Understanding Encoder
   - 6.7 Background Music Generation
   - 6.8 Singing Synthesis
   - 6.9 Vocal + Instrumental Mixing
   - 6.10 Song Composer Orchestration
7. [Full Control Layer](#7-full-control-layer-aarambh-voice-control)
8. [KV Cache & Streaming Inference](#8-kv-cache--streaming-inference)
9. [Training Pipeline](#9-training-pipeline)
10. [Custom Kernels](#10-custom-kernels-aarambh-voice-kernel)
11. [Quantisation](#11-quantisation-aarambh-voice-quant)
12. [Fine-Tuning](#12-fine-tuning-aarambh-voice-finetune)
13. [Safety Layer](#13-safety-layer-aarambh-voice-safety)
14. [Evaluation Harness](#14-evaluation-harness-aarambh-voice-eval)
15. [Crate-by-Crate Reference](#15-crate-by-crate-reference)
16. [Data Flow Across the Workspace](#16-data-flow-across-the-workspace)
17. [Memory & Compute Estimates](#17-memory--compute-estimates)
18. [Hardware Strategy](#18-hardware-strategy)
19. [Relationship to aarambh-ai](#19-relationship-to-aarambh-ai)
20. [What's Explicitly Out of Scope](#20-whats-explicitly-out-of-scope)

---

## 1. Project Overview

**aarambh-voice-studio** is a ground-up audio generation and understanding
system written entirely in Rust. Like `aarambh-ai`, it is not a wrapper
around PyTorch or any Python library — every codec, every conditioning
layer, every training loop is implemented from scratch using `candle-core`
and `candle-nn` as the tensor backend.

### What it is

- A **neural audio codec** that turns raw waveforms into discrete tokens and
  back — the same trick that lets `aarambh-ai`'s transformer generate audio
  the way it generates text.
- A **Voice Engine**: text-to-speech, zero-shot voice cloning from a few
  seconds of reference audio, text-described voice design (no reference
  audio needed), and continuous emotion control over the output.
- A **Music Engine**: text-to-instrumental generation, and a music
  understanding encoder that labels genre, tempo, key, mood, and
  instrumentation — used both as a standalone feature and to auto-label
  training data.
- A **Singing Engine**: lyrics + melody + duration → sung vocals, a cappella
  or mixed with a backing track, with the same cloning and emotion control
  as the Voice Engine.
- A **Song Composer**: orchestrates all three engines to turn lyrics and a
  style prompt into a finished song end-to-end.
- A **Full Control Layer**: one typed request struct that exposes every
  parameter above as an explicit, composable field — nothing is hidden
  behind presets you can't override.

### What makes it different

Most open speech/music tools are thin Rust bindings around a Python
inference script, or a single-purpose model with no shared infrastructure
across voice, music, and singing. aarambh-voice-studio builds the codec, the
conditioning layers, the training loop, the quantisation path, and the
control API once, in Rust, and reuses them across all four subsystems —
mirroring the crate discipline that `aarambh-ai` already proved out for
text.

### The four subsystems, one foundation

```
                    ┌───────────────────────────┐
                    │   Full control layer       │  ← one request struct
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │   Song composer             │  ← orchestrates below
                    └───┬───────────┬───────────┬─┘
                        │           │           │
                 ┌──────┴───┐ ┌─────┴────┐ ┌────┴──────┐
                 │  Voice    │ │  Music   │ │  Singing  │
                 │  engine   │ │  engine  │ │  engine   │
                 └──────┬────┘ └────┬─────┘ └────┬──────┘
                        └───────────┴─────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │   Shared foundation              │
                    │   neural audio codec +          │
                    │   transformer core              │
                    └──────────────────────────────────┘
```

---

## 2. Design Philosophy

| Goal | Decision |
|---|---|
| Reuse, don't rebuild | Transformer block (RMSNorm, RoPE, GQA, SwiGLU) is imported straight from `aarambh-ai-nn` patterns, not reinvented |
| Adaptation over from-scratch pretraining | Fine-tune/adapt open audio-codec backbones via LoRA/QLoRA/DoRA first; from-scratch pretraining is a stretch goal per subsystem, not the default path |
| One control surface | Every knob (voice, emotion, melody, mix) is a typed field on one request struct, never a hidden preset |
| Separate-then-mix over joint generation | Vocals and instrumentals are generated independently and mixed, not jointly modelled — more controllable, more debuggable |
| Understanding before generation | The music/emotion classifiers are built before the matching generator, both as a feature and as an auto-labelling tool for training data |
| No opaque cloning | Every voice-cloning path carries a consent flag and an inaudible watermark on generated output — see §13 |
| Source-first releases | Same policy as `aarambh-ai`: crates ship as source, `publish = false` until proven; no pretrained checkpoints bundled by default |
| i3 + free Kaggle GPU only | No paid compute assumed anywhere in the roadmap |

---

## 3. Dependency Versions & Toolchain

```toml
[workspace.dependencies]
# Tensor backend — shared with aarambh-ai, pin to the same minor version
candle-core        = { version = "0.10" }
candle-nn          = { version = "0.10" }
candle-transformers = { version = "0.10" }

# Audio I/O and DSP
hound              = "3"            # WAV read/write
symphonia          = { version = "0.5", features = ["all"] }  # decode mp3/flac/ogg input
rubato             = "0.16"         # resampling
rustfft            = "6"            # STFT/mel spectrogram
apodize            = "1"            # window functions for STFT

# Text / phonemes
tokenizers         = "0.21"
deunicode          = "1"            # transliteration fallback for Sanskrit/Hindi text prep

# Shared with aarambh-ai
anyhow             = "1"
thiserror          = "2"
serde              = { version = "1", features = ["derive"] }
serde_json         = "1"
toml               = "0.8"
tokio              = { version = "1", features = ["full"] }
clap               = { version = "4", features = ["derive"] }
tracing            = "0.1"
tracing-subscriber = "0.3"
safetensors        = "0.8"
rayon              = "1"
cc                 = "1"
which              = "6"
criterion          = "0.5"
sha2               = "0.10"

# Serving
axum               = "0.8"
```

> **Note on the codec:** the neural audio codec (encoder/decoder) is
> implemented from scratch in `aarambh-voice-codec` using `candle`, following
> a residual-vector-quantisation design (EnCodec/DAC family). It is not a
> binding to a Python codec library — same "build it in Rust" discipline as
> `aarambh-ai`'s tokenizer.

> **Per-crate Cargo.toml:** each crate's `[dependencies]` uses
> `workspace = true`. See §15 for the exact dependency list per crate.

---

## 4. Complete Workspace — 20 Library Crates

```
aarambh-voice-studio/
├── Cargo.toml                          ← [workspace] manifest
├── ARCHITECTURE_VOICE_STUDIO.md
├── ROADMAP_VOICE_STUDIO.md
│
├── crates/
│   │
│   │   ── LAYER 0: Shared types ──
│   ├── aarambh-voice-core/             Configs, request/response types, error types
│   │
│   │   ── LAYER 1: Foundation ──
│   ├── aarambh-voice-codec/            Neural audio codec (encode/decode, RVQ tokens)
│   ├── aarambh-voice-data/             Dataset loaders, preprocessing, auto-labelling
│   │
│   │   ── LAYER 2: Shared model primitives ──
│   ├── aarambh-voice-nn/               Transformer block + conditioning layers
│   ├── aarambh-voice-kernel/           CPU SIMD kernels, CUDA build prep
│   │
│   │   ── LAYER 3: Model definitions ──
│   ├── aarambh-voice-model/            AudioLM model definitions per engine
│   ├── aarambh-voice-weights/          SafeTensors save/load, checkpoint convert
│   │
│   │   ── LAYER 4: Training & compression ──
│   ├── aarambh-voice-train/            Pretraining / continued-training loops
│   ├── aarambh-voice-quant/            INT8 / INT4 / GGUF-style quantisation
│   │
│   │   ── LAYER 5: Voice Engine ──
│   ├── aarambh-voice-finetune/         LoRA / QLoRA / DoRA adapters (all engines)
│   ├── aarambh-voice-speaker/          Zero-shot cloning + text-described voice design
│   ├── aarambh-voice-emotion/          Emotion embedding space + intensity control
│   │
│   │   ── LAYER 6: Music Engine ──
│   ├── aarambh-voice-music/            Music understanding + background music generation
│   │
│   │   ── LAYER 7: Singing Engine ──
│   ├── aarambh-voice-sing/             Melody/duration-conditioned singing synthesis
│   ├── aarambh-voice-mix/              Vocal + instrumental mixing/mastering
│   │
│   │   ── LAYER 8: Composition ──
│   ├── aarambh-voice-compose/          Lyrics-to-song orchestrator
│   │
│   │   ── LAYER 9: Cross-cutting ──
│   ├── aarambh-voice-safety/           Consent gating, watermarking, misuse guardrails
│   ├── aarambh-voice-eval/             Evaluation harness (WER, speaker sim, music metrics)
│   ├── aarambh-voice-control/          Full control API / unified request DSL
│   ├── aarambh-voice-inference/        Shared inference runtime, KV cache, streaming
│   │
│   │   ── LAYER 10: Serving ──
│   └── aarambh-voice-serve/            HTTP inference server, batching, streaming audio
│
└── aarambh-voice-studio/                ← LAYER 11: CLI binary
    └── src/cmd/
        ├── speak.rs        (TTS)
        ├── clone.rs        (voice cloning)
        ├── design.rs       (text-described voice)
        ├── music.rs        (generate / analyse)
        ├── sing.rs         (singing synthesis)
        ├── compose.rs      (lyrics-to-song)
        ├── train.rs / finetune.rs / quantise.rs / eval.rs
        └── serve.rs
```

### Crate Count

```
20 library crates + 1 binary = 21 total
(aarambh-ai has 16 library crates + 1 binary — aarambh-voice-studio is
 larger because audio has more independent subsystems: codec, three
 generation engines, mixing, and composition, none of which exist in the
 text-only project.)
```

---

## 5. Model Scales

Every engine shares the same four-scale convention as `aarambh-ai`, so a
Tiny voice model trains on the same i3 laptop a Tiny text model does.

| Scale | Params | d_model | n_layers | n_heads | Use |
|---|---|---|---|---|---|
| Tiny   | ~20M  | 320  | 6  | 5  | i3 smoke tests, all unit tests |
| Small  | ~110M | 512  | 10 | 8  | Kaggle T4, first real-quality checkpoints |
| Medium | ~340M | 768  | 16 | 12 | Kaggle P100, production voice quality |
| Large  | ~900M | 1024 | 24 | 16 | Kaggle A100 (if available), best quality |

Each subsystem (Voice, Music, Singing) reuses this exact scale table —
`ModelConfig::tiny()` / `::small()` / `::medium()` / `::large()` — with a
`domain: AudioDomain` field (`Speech | Music | Singing`) selecting which
conditioning heads are attached (see §6).

---

## 6. The Full Journey: Text/Audio → Output

### 6.1 Neural Audio Codec (Tokenising Sound)

Raw audio can't feed a transformer directly — a 10-second clip at 24kHz is
240,000 samples. `aarambh-voice-codec` compresses audio into a short
sequence of discrete tokens using **Residual Vector Quantisation (RVQ)**:

```
waveform ──► Conv1D encoder ──► continuous latent ──► RVQ (N codebooks)
                                                            │
                                                    discrete token grid
                                                    [n_codebooks × n_frames]
```

- The encoder downsamples audio (e.g. 24kHz → ~75 frames/sec).
- RVQ quantises each frame's latent against a cascade of codebooks — the
  first codebook captures coarse structure, later ones capture residual
  detail. Typical setup: 8 codebooks × 1024 entries each.
- The decoder is the mirror of the encoder — token grid back to waveform.
- Trained once with a reconstruction + adversarial + codebook-commitment
  loss, then **frozen** — every downstream engine (Voice, Music, Singing)
  treats the codec as a fixed tokenizer/detokenizer, exactly like
  `aarambh-ai-tokenizer`'s BPE tokenizer is fixed once trained.

This single decision is why the rest of the project can reuse
`aarambh-ai`'s transformer stack almost unchanged: once audio is tokens, a
decoder-only transformer generates audio tokens the same way it generates
text tokens.

### 6.2 Shared Transformer Core

`aarambh-voice-nn` ports the exact block used in `aarambh-ai-nn`
(`ARCHITECTURE.md` §6): RMSNorm, RoPE, Grouped-Query Attention, SwiGLU,
pre-norm residual layout. The only addition is a **conditioning injection
point** after the embedding layer:

```
audio_tokens ──► embed ──► [+ speaker_emb] ──► [+ emotion_emb] ──► [+ style_emb]
                                                                        │
                                                              transformer blocks
                                                                        │
                                                              LM head → next audio token
```

Each conditioning embedding is added (or cross-attended, for longer
conditioning sequences like melody contours) at every layer, not just the
input — the same "conditioning must survive depth" lesson learned from
`aarambh-ai`'s thinking-engine budget injection.

### 6.3 Speaker Conditioning — Zero-Shot Voice Cloning

`aarambh-voice-speaker` implements a **speaker encoder**: a small
convolutional/transformer network that takes 3–10 seconds of reference
audio and produces a fixed-size speaker embedding vector (e.g. 256-dim).
That vector conditions the Voice/Singing transformer at generation time —
no per-speaker training required.

```
reference audio ──► speaker encoder ──► speaker_embedding (256-dim)
                                              │
                                    fed into every transformer layer
                                    as an additive conditioning vector
```

A **few-shot fine-tune path** also exists for higher fidelity on a specific
speaker: LoRA-adapt the transformer directly on that speaker's data,
reusing `aarambh-voice-finetune` — same infra as `aarambh-ai`'s LoRA/QLoRA.

### 6.4 Voice Design — Text-Described Voice

No reference audio needed. A small text encoder maps a description
("deep male voice, warm, slight rasp, 40s") to a point in the same
speaker-embedding space used for cloning:

```
"deep male voice, warm, slight rasp" ──► text encoder ──► speaker_embedding
                                                                │
                                                same conditioning path as §6.3
```

Trained on (description, speaker_embedding) pairs generated by captioning
existing speaker embeddings — cheap to train because it's a small
embedding-to-embedding regression, not a new generative model.

### 6.5 Emotion Conditioning

`aarambh-voice-emotion` owns an **emotion embedding space** with one
learned vector per discrete emotion (joy, sadness, anger, fear, calm,
excitement, and others), plus a **continuous intensity control**:

```
emotion_vector = base_neutral + intensity × (emotion_target − base_neutral)
```

`intensity ∈ [0.0, 1.0]` per emotion dimension, and multiple emotions can be
blended by weighted-summing their target vectors before interpolating —
giving sliders instead of on/off toggles, per the full-control requirement
in §7. The exact same mechanism conditions both the Voice Engine and the
Singing Engine (§6.8) — one emotion system, two consumers.

### 6.6 Music Understanding Encoder

`aarambh-voice-music` first builds an **understanding** path: an audio
encoder + classification heads predicting genre, tempo (BPM), musical key,
mood, instrumentation, and energy level from an input clip. Two uses:

1. **Standalone feature** — "what genre/tempo/key is this clip?"
2. **Auto-labelling** for training data — label large unlabelled music
   corpora automatically instead of hand-labelling, mirroring
   `aarambh-ai`'s automated dataset-download-and-label pipeline discipline.

### 6.7 Background Music Generation

Text prompt ("lo-fi hip-hop, rainy, 80 BPM") → instrumental track. A text
encoder embedding conditions the same transformer-over-audio-tokens
architecture as speech, just without phoneme/speaker conditioning:

```
text prompt ──► text encoder ──► style_embedding
                                       │
                          transformer over audio codec tokens
                                       │
                              codec decoder ──► instrumental waveform
```

This is the heaviest phase compute-wise — music has more acoustic
variability per second than speech, so adapting an existing open backbone
via LoRA/QLoRA is the realistic default path; from-scratch pretraining is a
stretch goal, not the plan.

### 6.8 Singing Synthesis

Needs three conditioning inputs instead of one:

```
lyrics (text) ──► phonemes ──┐
melody (pitch contour)  ─────┼──► transformer over audio codec tokens
per-syllable duration    ────┘         [+ speaker_emb] [+ emotion_emb]
                                             │
                                   codec decoder ──► sung waveform
```

Pitch is now an explicit, controllable input rather than something the
model infers on its own — the key structural difference from TTS. If no
melody is supplied, `aarambh-voice-compose` (§6.10) generates one.

### 6.9 Vocal + Instrumental Mixing

`aarambh-voice-mix` combines singing output with a backing track:

- **Alignment** — tempo/beat-match the vocal to the backing track.
- **Gain staging** — independent vocal/music gain, sidechain ducking so
  vocals sit forward in the mix.
- **Mastering pass** — simple loudness normalisation (LUFS target) and a
  light limiter.

Vocals and instrumentals are generated **separately, then mixed** — not
jointly modelled — per the design philosophy in §2: more controllable
(swap the backing track independently) and easier to debug than a single
joint model.

### 6.10 Song Composer Orchestration

`aarambh-voice-compose` is a **pipeline, not a new model**. Given lyrics and
a style prompt, it:

1. Generates a melody/pitch contour if none is supplied (a small
   sequence model over note/duration pairs, conditioned on lyrics'
   syllable structure and the style prompt).
2. Calls the Music Engine (§6.7) for the backing track.
3. Calls the Singing Engine (§6.8) for the vocals, conditioned on the
   melody from step 1.
4. Calls the Mixer (§6.9) to combine them.

Because subsystems 1–3 already exist by the time this phase is built, the
composer is integration risk, not modelling risk.

---

## 7. Full Control Layer (`aarambh-voice-control`)

One typed request struct is the single source of truth for every knob in
the system — CLI, library API, and the future inference server (§18 of the
roadmap) are all thin layers over this struct:

```rust
pub struct NaadRequest {
    pub content: Content,                       // Text(String) | Lyrics(String)
    pub voice: VoiceSpec,                        // Cloned(embedding) | Designed(desc) | Preset(id)
    pub emotion: EmotionSpec,                    // per-emotion intensity map, default neutral
    pub singing: Option<SingingSpec>,            // melody, tempo, key — None = spoken, not sung
    pub background_music: Option<MusicSpec>,     // genre, mood, instruments, tempo
    pub mix: MixSpec,                            // vocal_gain, music_gain, target_lufs
    pub consent_token: Option<ConsentToken>,     // required for non-preset cloning, see §13
}
```

Every field is explicit and typed — there is no hidden "smart default" that
can't be overridden. `aarambh-voice-control` validates a request (e.g.
cloning requires a consent token), resolves it into the lower-level calls
across `aarambh-voice-speaker` / `aarambh-voice-emotion` /
`aarambh-voice-music` / `aarambh-voice-sing` / `aarambh-voice-mix`, and
returns a single `NaadResponse { audio, metadata }`.

---

## 8. KV Cache & Streaming Inference

Audio token sequences are long relative to text (a 10-second clip at 75
codec frames/sec × 8 codebooks is thousands of tokens), so streaming
inference matters more here than in `aarambh-ai`:

- KV cache preallocated per request at `max_frames × n_codebooks`, same
  fixed-capacity pattern as `aarambh-ai`'s long-context KV cache
  (`ARCHITECTURE_V2.md` §21).
- Codec tokens are generated frame-by-frame; each completed frame is
  immediately decoded through the codec's decoder and streamed to the
  caller — the user hears audio well before generation finishes, the same
  latency-hiding trick TTS products rely on.
- `aarambh-voice-inference` owns this shared runtime; `aarambh-voice-serve`
  (§ roadmap Phase 22) exposes it over HTTP with chunked audio responses.

---

## 9. Training Pipeline

```
Stage 0 — Codec pretraining (once, frozen afterward)
Stage 1 — Domain pretraining per engine (Speech / Music / Singing token modelling)
Stage 2 — Conditioning fine-tune (speaker, emotion, melody heads)
Stage 3 — Task-specific adapters (LoRA/QLoRA/DoRA — see §12)
```

Same AdamW + cosine schedule + checkpointing discipline as
`aarambh-ai-train`. The key difference: Stage 0 (codec) is trained once and
never touched again, while Stages 1–3 are repeated per engine and per
domain, sharing the same training loop code with a `domain` and
`conditioning` config switch rather than three separate training binaries.

---

## 10. Custom Kernels (`aarambh-voice-kernel`)

Same CPU SIMD + CUDA-prep discipline as `aarambh-ai-kernel`
(`ARCHITECTURE.md` §10): AVX2/FMA fused RMSNorm, parallel attention via
`rayon` on CPU, with CUDA feature-gated kernels (Flash Attention v2, fused
RMSNorm/RoPE/SwiGLU) built later once the model shape is stable. Audio
adds one additional kernel target: a fused STFT/mel-spectrogram routine
used by the codec encoder and the music-understanding front end.

---

## 11. Quantisation (`aarambh-voice-quant`)

Identical ladder to `aarambh-ai-quant`: INT8 → GPTQ/AWQ INT4 → GGUF-style
packed format, plus optional QAT. Applied to the transformer core in every
engine; the codec encoder/decoder is small enough to usually stay in F32
or INT8 only. Quantised checkpoints matter more here than for text because
a real-time streaming voice product on modest hardware needs low-latency
CPU inference.

---

## 12. Fine-Tuning (`aarambh-voice-finetune`)

LoRA, QLoRA, and DoRA — same weight-decomposed adapter math as
`aarambh-ai-finetune` (`ARCHITECTURE_V2.md` §23) — applied across every
engine:

| Use | Adapter target |
|---|---|
| Per-speaker high-fidelity cloning | Attention + FFN LoRA on Voice Engine |
| Style/genre specialisation | LoRA on Music Engine |
| Singing style (e.g. specific singer's technique) | DoRA on Singing Engine |
| Language/accent adaptation | LoRA on shared transformer core |

Reusing one adapter implementation across four consumers is the same
"build the mechanism once" discipline as §6.3/§6.5's shared conditioning
embeddings.

---

## 13. Safety Layer (`aarambh-voice-safety`)

Voice cloning and singing-voice cloning carry real misuse risk, so this
layer is not optional, unlike in `aarambh-ai` where safety was primarily
about text content:

- **Consent gating** — non-preset voice cloning requires a `ConsentToken`
  attached to the reference audio (a recorded or typed consent statement,
  or a signed developer-side attestation for pre-cleared voices).
- **Inaudible watermarking** — every generated waveform (speech, singing,
  and music) is watermarked with an inaudible signal identifying it as
  synthetic, decodable by a companion detector.
- **Misuse guardrails** — text/lyrics content guardrails reuse
  `aarambh-ai-safety`'s input/output guardrail pattern directly (PII
  detection, disallowed-content filtering) applied to the text/lyrics
  input before synthesis.
- **Audit logging** — every cloning request logs a hash of the reference
  audio and the consent token, not the raw audio itself.

---

## 14. Evaluation Harness (`aarambh-voice-eval`)

Perplexity alone doesn't tell you if a voice sounds right, so this harness
adds domain-appropriate proxy metrics, mirroring `aarambh-ai-eval`'s
"more than one signal" philosophy (`ARCHITECTURE_V2.md` §22):

| Task | What it measures | Scoring |
|---|---|---|
| ASR round-trip WER | TTS intelligibility | Feed generated speech through an ASR model, compare to source text |
| Speaker similarity | Cloning fidelity | Cosine similarity between reference and generated speaker embeddings |
| Emotion classification accuracy | Emotion control fidelity | Run an emotion classifier on generated output, compare to requested emotion |
| Music tag agreement | Generation-to-prompt fidelity | Run the music understanding encoder (§6.6) on generated output, compare tags to the prompt |
| MOS proxy | Overall naturalness | A learned quality predictor trained on public MOS-labelled datasets |

---

## 15. Crate-by-Crate Reference

| Crate | Layer | Responsibility |
|---|---|---|
| `aarambh-voice-core` | 0 | Shared config types, request/response types, errors |
| `aarambh-voice-codec` | 1 | RVQ neural audio codec, encode/decode |
| `aarambh-voice-data` | 1 | Dataset loaders, preprocessing, auto-labelling pipelines |
| `aarambh-voice-nn` | 2 | Transformer block + conditioning injection layers |
| `aarambh-voice-kernel` | 2 | CPU SIMD kernels, CUDA build prep, fused STFT |
| `aarambh-voice-model` | 3 | Per-engine model definitions built on `-nn` |
| `aarambh-voice-weights` | 3 | SafeTensors save/load, checkpoint conversion |
| `aarambh-voice-train` | 4 | Pretraining and continued-training loops |
| `aarambh-voice-quant` | 4 | INT8 / INT4 / GGUF-style quantisation |
| `aarambh-voice-finetune` | 5 | LoRA / QLoRA / DoRA adapters, all engines |
| `aarambh-voice-speaker` | 5 | Zero-shot cloning, text-described voice design |
| `aarambh-voice-emotion` | 5 | Emotion embedding space, continuous intensity |
| `aarambh-voice-music` | 6 | Music understanding + background music generation |
| `aarambh-voice-sing` | 7 | Singing synthesis |
| `aarambh-voice-mix` | 7 | Vocal + instrumental mixing/mastering |
| `aarambh-voice-compose` | 8 | Lyrics-to-song orchestrator |
| `aarambh-voice-safety` | 9 | Consent gating, watermarking, guardrails |
| `aarambh-voice-eval` | 9 | Evaluation harness |
| `aarambh-voice-control` | 9 | Full control API / unified request DSL |
| `aarambh-voice-inference` | 9 | Shared inference runtime, KV cache, streaming |
| `aarambh-voice-serve` | 10 | HTTP inference server |
| `aarambh-voice-studio` (bin) | 11 | CLI |

---

## 16. Data Flow Across the Workspace

```
raw audio / text / lyrics
        │
        ▼
aarambh-voice-data ──► preprocessing, auto-labelling (via aarambh-voice-music understanding)
        │
        ▼
aarambh-voice-codec ──► discrete audio tokens (frozen after Stage 0 training)
        │
        ▼
aarambh-voice-nn + aarambh-voice-model ──► conditioned transformer
        │            ▲            ▲            ▲
        │      speaker_emb   emotion_emb   melody/duration
        │    (aarambh-voice-  (aarambh-voice-  (aarambh-voice-
        │     speaker)         emotion)          sing)
        ▼
aarambh-voice-inference ──► streamed audio tokens ──► codec decode ──► waveform
        │
        ▼
aarambh-voice-mix (if singing + music) ──► aarambh-voice-safety (watermark) ──► output
```

`aarambh-voice-control` sits above this whole flow, translating one
`NaadRequest` into the correct sequence of calls; `aarambh-voice-compose`
sits above `-control` for the full lyrics-to-song path.

---

## 17. Memory & Compute Estimates

### Training Memory (BF16, per scale, transformer core only)

| Scale | Weights | Gradients | AdamW States | Activations | Total |
|---|---|---|---|---|---|
| Tiny   | 40 MB   | 40 MB   | 160 MB  | ~80 MB  | ~0.32 GB |
| Small  | 220 MB  | 220 MB  | 880 MB  | ~440 MB | ~1.76 GB |
| Medium | 680 MB  | 680 MB  | 2.7 GB  | ~1.4 GB | ~5.5 GB  |
| Large  | 1.8 GB  | 1.8 GB  | 7.2 GB  | ~2.9 GB | ~13.7 GB |

> Codec encoder/decoder adds a roughly fixed ~150–300 MB regardless of
> transformer scale, and is trained separately (Stage 0) — see §9.

### CPU Inference Memory (F32 weights + KV cache, 10s output)

| Scale | Weights | KV Cache | Codec | Total |
|---|---|---|---|---|
| Tiny   | 80 MB   | 12 MB  | ~150 MB | ~242 MB |
| Small  | 440 MB  | 40 MB  | ~150 MB | ~630 MB |
| Medium | 1.36 GB | 160 MB | ~200 MB | ~1.72 GB |
| Large  | 3.6 GB  | 320 MB | ~250 MB | ~4.17 GB |

### CPU Inference Memory (INT4 quantised weights + KV cache, 10s output)

| Scale | Weights (Q4) | KV Cache | Codec | Total |
|---|---|---|---|---|
| Tiny   | 11 MB  | 12 MB  | ~150 MB | ~173 MB |
| Small  | 58 MB  | 40 MB  | ~150 MB | ~248 MB |
| Medium | 178 MB | 160 MB | ~200 MB | ~538 MB |
| Large  | 470 MB | 320 MB | ~250 MB | ~1.04 GB |

---

## 18. Hardware Strategy

### Your Local Machine (i3-1115G4, 8 GB RAM, Pop OS)

**Use exclusively for Tiny scale**, exactly as with `aarambh-ai`:
- Codec smoke-training on a small public speech subset (LJSpeech excerpt)
- Full Tiny TTS training loop
- All unit and integration tests
- CLI inference (`speak`, `clone`, `design`) with Tiny checkpoints
- QLoRA fine-tuning of Small on a single speaker's data
- INT4 inference of Medium

**Recommended Tiny training config (Voice Engine, Stage 1):**
```toml
batch_size        = 2
grad_accum_steps  = 16
max_audio_seconds = 4          # short clips, saves RAM
dataset           = "data/ljspeech_subset/"
learning_rate     = 1e-3
max_steps         = 5000
warmup_steps      = 200
device            = "cpu"
dtype             = "f32"
eval_steps        = 500
```

### Kaggle GPU

| Scale | GPU | Dtype | Batch | Notes |
|---|---|---|---|---|
| Small  | T4 16 GB   | BF16 | 8  | Codec + Voice Engine Stage 1/2 |
| Medium | P100 16 GB | BF16 | 4  | Music Engine generation (heaviest phase) |
| Large  | A100 40 GB | BF16 | 2  | Stretch goal, hardware-dependent |

Same opt-in `--features cuda` pattern as `aarambh-ai`: CPU builds remain
the default, GPU is switched on per training run via config
(`device = "cuda:0"`, `dtype = "bf16"`).

### Which phases actually need Kaggle

| Subsystem | Realistic on i3 alone? |
|---|---|
| Voice Engine (TTS, cloning, design, emotion) | Yes, for Tiny/Small; Kaggle for Medium+ |
| Music Engine — understanding | Yes on i3 |
| Music Engine — generation | No — needs Kaggle, this is the heaviest phase |
| Singing Engine | Kaggle recommended, i3 only for Tiny smoke tests |
| Composer + Control + Mix | Yes on i3 — integration work, not training |

---

## 19. Relationship to `aarambh-ai`

aarambh-voice-studio is a **sibling project**, not a fork:

- The transformer block in `aarambh-voice-nn` is the same architecture as
  `aarambh-ai-nn`, ported rather than shared as a direct crate dependency —
  audio conditioning needs injection points text generation doesn't, so the
  two are kept decoupled to avoid coupling audio changes to text-model
  releases.
- Training, fine-tuning, quantisation, and safety-layer discipline are all
  directly modelled on `aarambh-ai`'s equivalent crates — same phase
  structure, same "source release, no bundled weights" policy, same
  i3-first hardware strategy.
- The two projects could eventually share tokenised "modalities" in one
  unified multimodal model (text + audio in one transformer), but that is
  explicitly out of scope for `aarambh-voice-studio` v1 — see §20.

---

## 20. What's Explicitly Out of Scope (v1)

- Real-time (sub-100ms) streaming synthesis — first release targets
  chunked streaming (§8), not conversational-latency voice.
- Joint text+audio multimodal model combining `aarambh-ai` and
  aarambh-voice-studio into one transformer.
- Video or lip-sync generation.
- From-scratch pretraining of the Music Engine as the default path —
  adaptation of an open backbone is the plan; from-scratch is a stretch
  goal only (§2, §18).
- Multi-GPU distributed training (single-GPU Kaggle only, matching
  `aarambh-ai` v1.0's own scope before its v2.0 multi-GPU phase).
- Pretrained checkpoints, adapters, or voice packs bundled in the
  repository — source/engineering release only, same policy as
  `aarambh-ai`.
