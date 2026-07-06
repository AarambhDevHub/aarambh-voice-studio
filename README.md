# aarambh-voice-studio

> Sanskrit: Aarambh means beginning. Naad means sound.  
> A Rust-native AI audio studio for speech, music, singing, and full-song creation.

![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange)
![Framework](https://img.shields.io/badge/AI%20Backend-Candle-blue)
![Status](https://img.shields.io/badge/Status-Roadmap%20%2F%20Engineering%20Build-yellow)
![Release](https://img.shields.io/badge/Release-Source%20Only-lightgrey)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

`aarambh-voice-studio` is a from-scratch speech, music, and singing engine written in Rust using Candle. It is designed as a sibling project to [`aarambh-ai`](https://github.com/AarambhDevHub/aarambh-ai), but instead of text tokens only, it builds a shared neural-audio-codec transformer core for audio generation and understanding.

The goal is not only text-to-speech. The goal is a full AI audio studio:

- create spoken voice from text
- clone a voice with consent
- design a voice from a text description
- control emotion and intensity
- understand and generate background music
- synthesize singing from lyrics and melody
- mix vocals with instrumental music
- compose a full song from lyrics and style prompts

This repository is a **source-first engineering project**. It does not ship pretrained checkpoints, voice packs, cloned voices, adapters, or generated voice assets by default.

Inspired by: VALL-E · AudioLM · MusicGen · EnCodec · DAC · Bark · Stable Audio · LLaMA · DeepSeek · Mistral · aarambh-ai

---

## Project Status

`aarambh-voice-studio` is currently a roadmap-stage engineering build. The architecture and roadmap define the system, crate layout, training plan, and release policy. Implementation should follow the phase order in `ROADMAP_VOICE_STUDIO.md`.

| Area | Status |
|---|---|
| Rust-only architecture | Planned |
| Candle tensor backend | Planned |
| 20 library crates + 1 CLI binary | Planned |
| Neural audio codec | Planned |
| TTS baseline | Planned |
| Voice cloning | Planned |
| Emotion control | Planned |
| Music understanding | Planned |
| Background music generation | Planned |
| Singing synthesis | Planned |
| Lyrics-to-song composer | Planned |
| Safety and watermarking | Planned |
| Quantisation and evaluation | Planned |
| HTTP inference server | Planned |

Do not treat this repository as a finished production model until the relevant roadmap phases are implemented, tested, and tagged.

---

## What Makes This Project Different

Most audio AI tools are either Python-first, single-purpose, or thin Rust wrappers around external inference scripts. `aarambh-voice-studio` is designed differently:

1. **Rust-native** — model code, training loops, codec, inference, safety, and CLI are implemented in Rust.
2. **Candle-based** — `candle-core`, `candle-nn`, and `candle-transformers` are used as the tensor and neural network foundation.
3. **One shared foundation** — voice, music, and singing reuse the same neural audio codec and transformer primitives.
4. **Three engines, one song composer** — Voice Engine, Music Engine, and Singing Engine work together to create full songs.
5. **Full control layer** — voice, emotion, music, melody, mix, and consent fields are exposed as typed request parameters.
6. **Source-first release policy** — no bundled pretrained weights, voice packs, cloned voices, or adapters.
7. **Safety is part of the architecture** — consent gating and watermarking are planned as core system features, not later add-ons.

---

## The Three Engines

### 1. Voice Engine

The Voice Engine handles spoken audio:

- text-to-speech
- zero-shot voice cloning from a short reference clip
- text-described voice design
- speaker embedding control
- emotion-controlled speech
- streaming inference

Example target command:

```bash
cargo run --release -p aarambh-voice-studio -- speak \
  --text "Hello from Aarambh Voice Studio" \
  --voice preset:neutral \
  --emotion calm:0.8 \
  --out hello.wav
```

### 2. Music Engine

The Music Engine handles music understanding and background music generation:

- genre detection
- tempo/BPM prediction
- key and mood classification
- instrumentation tagging
- text-to-instrumental generation
- auto-labelling for music datasets

Example target command:

```bash
cargo run --release -p aarambh-voice-studio -- music generate \
  --prompt "lo-fi hip-hop, rainy, 80 bpm" \
  --duration 30 \
  --out beat.wav
```

### 3. Singing Engine

The Singing Engine turns lyrics, melody, timing, speaker identity, and emotion into sung vocals:

- lyrics-to-phoneme alignment
- melody and pitch conditioning
- per-syllable duration control
- a cappella singing synthesis
- cloned singing voice support
- emotional singing

Example target command:

```bash
cargo run --release -p aarambh-voice-studio -- sing \
  --lyrics lyrics.txt \
  --melody melody.json \
  --voice designed:"bright expressive female" \
  --emotion joy:0.7 \
  --out vocal.wav
```

---

## Full Song Creation

The biggest idea in this project is that the three engines can work together.

```text
User input
  ├── lyrics
  ├── music style prompt
  ├── voice choice or voice description
  ├── emotion controls
  ├── optional melody / tempo / key
  └── mix controls
        ↓
Song Composer
  ├── create or resolve melody
  ├── generate backing music with Music Engine
  ├── generate sung vocals with Singing Engine
  ├── apply speaker and emotion controls from Voice Engine
  ├── mix vocals and music
  └── watermark final audio
        ↓
Finished song WAV/audio file
```

Example target command:

```bash
cargo run --release -p aarambh-voice-studio -- compose \
  --lyrics song.txt \
  --style "upbeat pop, 120 bpm, bright synths" \
  --voice designed:"clear young male, energetic" \
  --emotion excitement:0.8 \
  --out finished_song.wav
```

---

## Full Control API

All CLI commands are planned to become thin wrappers over one typed request system:

```rust
pub struct NaadRequest {
    pub content: Content,
    pub voice: VoiceSpec,
    pub emotion: EmotionSpec,
    pub singing: Option<SingingSpec>,
    pub background_music: Option<MusicSpec>,
    pub mix: MixSpec,
    pub consent_token: Option<ConsentToken>,
}
```

This is the core product idea: users should not be locked behind hidden presets. They should be able to control:

- text or lyrics
- voice preset, cloned voice, or designed voice
- emotion type and intensity
- melody, pitch, duration, key, and tempo
- background music style
- instruments and mood
- vocal gain and music gain
- loudness target
- consent metadata for cloning
- safety and watermarking behavior

---

## Architecture Overview

```text
                    ┌───────────────────────────┐
                    │   Full Control Layer       │
                    │   NaadRequest API          │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │   Song Composer            │
                    └───┬───────────┬───────────┬─┘
                        │           │           │
                 ┌──────┴───┐ ┌─────┴────┐ ┌────┴──────┐
                 │  Voice    │ │  Music   │ │  Singing  │
                 │  Engine   │ │  Engine  │ │  Engine   │
                 └──────┬────┘ └────┬─────┘ └────┬──────┘
                        └───────────┴─────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ Shared Neural Audio Foundation │
                    │ RVQ codec + Transformer core   │
                    └────────────────────────────────┘
```

---

## Workspace Layout

```text
aarambh-voice-studio/
├── Cargo.toml
├── README.md
├── ARCHITECTURE_VOICE_STUDIO.md
├── ROADMAP_VOICE_STUDIO.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
│
├── crates/
│   ├── aarambh-voice-core/          # Configs, request/response types, errors
│   ├── aarambh-voice-codec/         # Neural audio codec, RVQ tokens, decode
│   ├── aarambh-voice-data/          # Dataset loaders and preprocessing
│   ├── aarambh-voice-nn/            # Transformer blocks and conditioning
│   ├── aarambh-voice-kernel/        # CPU SIMD, CUDA prep, STFT kernels
│   ├── aarambh-voice-model/         # Model definitions per engine
│   ├── aarambh-voice-weights/       # SafeTensors save/load
│   ├── aarambh-voice-train/         # Training loops
│   ├── aarambh-voice-quant/         # INT8, INT4, GGUF-style quantisation
│   ├── aarambh-voice-finetune/      # LoRA, QLoRA, DoRA
│   ├── aarambh-voice-speaker/       # Voice cloning and voice design
│   ├── aarambh-voice-emotion/       # Emotion embeddings and intensity control
│   ├── aarambh-voice-music/         # Music understanding and generation
│   ├── aarambh-voice-sing/          # Singing synthesis
│   ├── aarambh-voice-mix/           # Vocal + instrumental mixing
│   ├── aarambh-voice-compose/       # Lyrics-to-song composer
│   ├── aarambh-voice-safety/        # Consent, watermarking, guardrails
│   ├── aarambh-voice-eval/          # WER, speaker sim, music metrics, MOS proxy
│   ├── aarambh-voice-control/       # Unified request API
│   ├── aarambh-voice-inference/     # KV cache and streaming inference
│   └── aarambh-voice-serve/         # HTTP server
│
└── aarambh-voice-studio/            # CLI binary
    └── src/cmd/
        ├── speak.rs
        ├── clone.rs
        ├── design.rs
        ├── music.rs
        ├── sing.rs
        ├── compose.rs
        ├── train.rs
        ├── finetune.rs
        ├── quantise.rs
        ├── eval.rs
        └── serve.rs
```

---

## Model Scales

| Scale | Approx Params | d_model | Layers | Heads | Target Use |
|---|---:|---:|---:|---:|---|
| Tiny | ~20M | 320 | 6 | 5 | i3 smoke tests and unit tests |
| Small | ~110M | 512 | 10 | 8 | Kaggle T4 first real checkpoint |
| Medium | ~340M | 768 | 16 | 12 | Production-quality voice target |
| Large | ~900M | 1024 | 24 | 16 | Best-quality research target |

Every subsystem uses the same scale pattern with `AudioDomain::Speech`, `AudioDomain::Music`, or `AudioDomain::Singing`.

---

## Quick Start

### Prerequisites

- Rust stable, 1.80 or later
- Git
- Linux recommended for development
- No GPU required for Phase 0 and CPU smoke tests
- Kaggle GPU or CUDA-capable machine recommended for serious codec, music, and singing training

### Clone

```bash
git clone https://github.com/AarambhDevHub/aarambh-voice-studio.git
cd aarambh-voice-studio
```

### Build and Test

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

### Build CLI

```bash
cargo build --release -p aarambh-voice-studio
cargo run --release -p aarambh-voice-studio -- --help
```

---

## Phase Roadmap

| Phase | Goal | Hardware |
|---:|---|---|
| 0 | Workspace + core types | i3 |
| 1 | Neural audio codec | i3 + Kaggle |
| 2 | Data pipeline + auto-labelling | i3 |
| 3 | NN primitives + conditioning injection | i3 |
| 4 | CPU SIMD kernels + CUDA prep | i3 + Kaggle prep |
| 5 | TTS baseline — Tiny trains | i3 + Kaggle |
| 6 | Inference engine + CLI | i3 |
| 7 | Voice cloning | Kaggle |
| 8 | Text-described voice design | i3 + Kaggle |
| 9 | Emotion control | i3 + Kaggle |
| 10 | Music understanding | i3 + Kaggle |
| 11 | Background music generation | Kaggle |
| 12 | Singing synthesis | Kaggle |
| 13 | Singing + music mixing | i3 + Kaggle |
| 14 | Cloning + emotion for singing | Kaggle |
| 15 | Lyrics-to-song composer | i3 + Kaggle |
| 16 | Full control layer | i3 |
| 17 | Safety and watermarking | i3 |
| 18 | Quantisation stack | i3 + Kaggle |
| 19 | Fine-tuning refinement | Kaggle |
| 20 | Evaluation harness | i3 + Kaggle |
| 21 | GPU scale-up | Kaggle |
| 22 | Inference server | i3 |
| 23 | Production release v1.0 | all |

See `ROADMAP_VOICE_STUDIO.md` for the full checklist.

---

## Safety Policy Summary

Voice generation and singing-voice cloning can be misused. This project therefore treats safety as a core engineering requirement.

Planned safety rules:

- non-preset voice cloning requires a consent token
- generated audio is watermarked
- reference audio is never stored in audit logs, only hashes
- text and lyrics pass through guardrails
- generated content must not impersonate people without permission
- cloned voice examples, speaker packs, and adapters are not bundled by default

See `SECURITY.md` for vulnerability reporting and security scope.

---

## Source Release Policy

This project follows the same source-first discipline as `aarambh-ai`:

- Build from repository source.
- Crates are not published to crates.io until stable.
- `publish = false` should be used during early phases.
- No pretrained checkpoints are included.
- No model weights, voice packs, adapters, cloned voices, or generated datasets are attached to releases.
- Example configs are for local smoke tests and user-created checkpoints.
- CUDA is optional; default CPU builds must remain valid.

---

## Contributing

Contributions are welcome, but this is a complex safety-sensitive audio project. Please read `CONTRIBUTING.md` before opening a pull request.

Good first areas:

- documentation fixes
- Phase 0 workspace scaffolding
- type definitions in `aarambh-voice-core`
- tests for config serialization
- WAV read/write utilities
- safe DSP helpers
- CLI help text

Avoid opening large modelling PRs without an issue first.

---

## License

Apache License 2.0. See `LICENSE`.

---

## Author

Created by **Darshan Vichhi** under **AarambhDevHub**.

