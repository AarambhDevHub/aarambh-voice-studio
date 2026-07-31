# aarambh-voice-studio

> Sanskrit: Aarambh means beginning. Naad means sound.
> A Rust-native AI audio studio for speech, music, singing, and full-song creation.

![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange)
![Framework](https://img.shields.io/badge/AI%20Backend-Candle-blue)
![Status](https://img.shields.io/badge/Status-Roadmap%20%2F%20Engineering%20Build-yellow)
![Release](https://img.shields.io/badge/Release-Source%20Only-lightgrey)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)
![Crates](https://img.shields.io/badge/Crates-24-informational)

`aarambh-voice-studio` is a from-scratch speech, music, and singing engine written in Rust using Candle. It is designed as a sibling project to [`aarambh-studio`](https://github.com/AarambhDevHub/aarambh-studio), but instead of text tokens only, it builds a shared neural-audio-codec transformer core for audio generation and understanding.

The goal is not only text-to-speech. The goal is a full AI audio studio:

- create spoken voice from text
- clone a voice with consent
- design a voice from a text description
- control emotion and intensity
- understand and generate background music
- synthesize singing from lyrics and melody, with an optional diffusion refinement pass
- mix vocals with instrumental music
- plan song structure (verse/chorus/bridge) before composing
- compose a full song from lyrics and style prompts
- align generation quality with GRPO/DPO, using its own evaluation metrics as rewards
- learn new voices and styles online after deployment, without a full retrain

This repository is a **source-first engineering project**. It does not ship pretrained checkpoints, voice packs, cloned voices, adapters, or generated voice assets by default.

Inspired by: VALL-E · AudioLM · MusicGen · EnCodec · DAC · Mimi · Bark · Stable Audio · LLaMA · DeepSeek · Mistral · aarambh-studio

---

## Project Status

`aarambh-voice-studio` is currently a roadmap-stage engineering build. The architecture and roadmap define the system, crate layout, training plan, and release policy. Implementation should follow the phase order in `ROADMAP_VOICE_STUDIO_PART1.md` / `ROADMAP_VOICE_STUDIO_PART2.md`.

| Area | Status |
|---|---|
| Rust-only architecture | Planned |
| Candle tensor backend | Planned |
| 23 library crates + 1 CLI binary | Planned |
| Neural audio codec (12.5Hz, transformer bottleneck, semantic distillation) | Planned |
| Text prep — G2P + normalisation | Planned |
| TTS baseline | Planned |
| Voice cloning | Planned |
| Voice design | Planned |
| Emotion control | Planned |
| Music understanding | Planned |
| Background music generation | Planned |
| Singing synthesis (autoregressive + optional diffusion refinement) | Planned |
| Structure planner + lyrics-to-song composer | Planned |
| Full control layer | Planned |
| Safety and watermarking | Planned |
| Quantisation | Planned |
| Fine-tuning (LoRA/QLoRA/DoRA) | Planned |
| Alignment (GRPO + DPO) | Planned |
| Self-learning (online, confidence-gated) | Planned |
| Evaluation harness + baseline comparison | Planned |
| Speculative decoding | Planned |
| HTTP inference server + multi-format output | Planned |

Do not treat this repository as a finished production model until the relevant roadmap phases are implemented, tested, and tagged.

---

## What Makes This Project Different

Most audio AI tools are either Python-first, single-purpose, or thin Rust wrappers around external inference scripts. `aarambh-voice-studio` is designed differently:

1. **Rust-native** — model code, training loops, codec, inference, safety, and CLI are implemented in Rust.
2. **Candle-based** — `candle-core`, `candle-nn`, and `candle-transformers` are used as the tensor and neural network foundation.
3. **One shared foundation** — voice, music, and singing reuse the same neural audio codec and transformer primitives.
4. **Three engines, one song composer** — Voice Engine, Music Engine, and Singing Engine work together, orchestrated by a structure planner that decides song shape before any audio is generated.
5. **Full control layer** — voice, emotion, music, melody, mix, output format, and consent fields are exposed as typed request parameters.
6. **Reward-aligned, not just loss-aligned** — GRPO and DPO refine quality using the project's own evaluation metrics as reward signals.
7. **Learns after it ships** — a self-learning subsystem absorbs new voices, styles, and corrections online, with confidence-gated commits and anti-forgetting guarantees, instead of requiring a full fine-tune job for every small update.
8. **Source-first release policy** — no bundled pretrained weights, voice packs, cloned voices, or adapters.
9. **Safety is part of the architecture** — consent gating and watermarking are core system features, not later add-ons.

---

## The Three Engines

### 1. Voice Engine

The Voice Engine handles spoken audio:

- text-to-speech
- zero-shot voice cloning from a short reference clip
- text-described voice design
- speaker embedding control
- emotion-controlled speech
- streaming inference with speculative decoding

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

The Singing Engine turns lyrics, melody, timing, speaker identity, and emotion into sung vocals, in two stages:

- **Stage A** — autoregressive lyrics-to-phoneme alignment, melody and pitch conditioning, per-syllable duration control, a cappella singing synthesis, cloned/emotional singing
- **Stage B (optional)** — a diffusion/flow-matching refinement pass on top of Stage A for extra naturalness, feature-gated `diffusion-refine`

Example target command:

```bash
cargo run --release -p aarambh-voice-studio -- sing \
  --lyrics lyrics.txt \
  --melody melody.json \
  --voice designed:"bright expressive female" \
  --emotion joy:0.7 \
  --refine \
  --out vocal.wav
```

---

## Full Song Creation

The biggest idea in this project is that the three engines can work together, with a structure planner deciding song shape first.

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
  ├── plan structure (verse / chorus / bridge, detect repeats)
  ├── create or resolve melody
  ├── generate backing music with Music Engine
  ├── generate sung vocals with Singing Engine (reusing audio for repeated choruses)
  ├── apply speaker and emotion controls from Voice Engine
  ├── mix vocals and music
  └── watermark final audio
        ↓
Finished song, in your chosen output format
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

## Alignment and Self-Learning

Two subsystems make quality a moving target the project actively optimizes against, rather than something fixed at training time:

**Alignment (`aarambh-voice-align`)** — GRPO and DPO training, using the same metrics the evaluation harness already computes (WER, speaker similarity, emotion accuracy, music tag agreement, MOS proxy) as reward signals. No separate reward model to train from scratch.

```bash
cargo run --release -p aarambh-voice-studio -- align \
  --engine tts --method dpo --scale small --out aligned.safetensors
```

**Self-learning (`aarambh-voice-selflearn`)** — online, confidence-gated adaptation for new voices and styles. Every update is staged, scored against the evaluation harness, and only committed if it doesn't regress quality — with gradient orthogonalization so learning a new voice never degrades one already known. Full design in `SELF_LEARNING_VOICE_STUDIO.md`.

```bash
cargo run --release -p aarambh-voice-studio -- learn \
  --sample new_voice.wav --identity-hint "warm, mid-30s"
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
    pub output_format: AudioOutputFormat,
    pub consent_token: Option<ConsentToken>,
    pub learn_from_this: bool,
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
- output audio format
- consent metadata for cloning
- whether this request should feed the self-learning system
- safety and watermarking behavior

---

## Audio Output Formats

| Format | Use case | Default? |
|---|---|---|
| WAV (PCM16/24/32f) | Lossless, universal compatibility, CLI default | **Yes, for CLI** |
| FLAC | Lossless, smaller than WAV | Available |
| Opus | Streaming server responses, lowest bandwidth | **Yes, for `serve`** |
| MP3 | Legacy compatibility | Off by default, behind the `mp3` cargo feature (LAME licensing) |

---

## Architecture Overview

```text
                    ┌────────────────────────────┐
                    │     Full Control Layer       │
                    │     NaadRequest API           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │   Song Composer                │
                    │   (structure planner first)     │
                    └───┬───────────┬───────────┬──┘
                        │           │           │
                 ┌──────┴───┐ ┌─────┴────┐ ┌────┴──────┐
                 │  Voice    │ │  Music   │ │  Singing  │
                 │  Engine   │ │  Engine  │ │  Engine   │
                 └──────┬────┘ └────┬─────┘ └────┬──────┘
                        └───────────┴─────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │  Shared Neural Audio Foundation    │
                    │  12.5Hz codec + Transformer core    │
                    │  + Alignment (GRPO/DPO)               │
                    │  + Self-Learning                       │
                    └──────────────────────────────────────┘
```

---

## Workspace Layout

```text
aarambh-voice-studio/
├── Cargo.toml
├── README.md
├── ARCHITECTURE_VOICE_STUDIO_PART1.md
├── ARCHITECTURE_VOICE_STUDIO_PART2.md
├── SELF_LEARNING_VOICE_STUDIO.md
├── ROADMAP_VOICE_STUDIO_PART1.md
├── ROADMAP_VOICE_STUDIO_PART2.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
│
├── crates/
│   ├── aarambh-voice-core/          # Configs, request/response types, errors
│   ├── aarambh-voice-codec/         # Neural audio codec, 12.5Hz, semantic distillation
│   ├── aarambh-voice-data/          # Dataset loaders and preprocessing
│   ├── aarambh-voice-textprep/      # G2P + text normalisation
│   ├── aarambh-voice-nn/            # Transformer blocks and conditioning
│   ├── aarambh-voice-kernel/        # CPU SIMD, CUDA prep, STFT kernels
│   ├── aarambh-voice-model/         # Model definitions per engine + diffusion refinement head
│   ├── aarambh-voice-weights/       # SafeTensors save/load
│   ├── aarambh-voice-train/         # Training loops
│   ├── aarambh-voice-quant/         # INT8, INT4, GGUF-style quantisation
│   ├── aarambh-voice-finetune/      # LoRA, QLoRA, DoRA
│   ├── aarambh-voice-align/         # GRPO + DPO alignment
│   ├── aarambh-voice-selflearn/     # Online self-learning, anti-forgetting
│   ├── aarambh-voice-speaker/       # Voice cloning and voice design
│   ├── aarambh-voice-emotion/       # Emotion embeddings and intensity control
│   ├── aarambh-voice-music/         # Music understanding and generation
│   ├── aarambh-voice-sing/          # Singing synthesis (AR + diffusion refinement)
│   ├── aarambh-voice-mix/           # Vocal + instrumental mixing
│   ├── aarambh-voice-compose/       # Structure planner + lyrics-to-song composer
│   ├── aarambh-voice-safety/        # Consent, watermarking, guardrails
│   ├── aarambh-voice-eval/          # WER, speaker sim, music metrics, MOS proxy, baseline comparison
│   ├── aarambh-voice-control/       # Unified request API
│   ├── aarambh-voice-inference/     # KV cache, streaming inference, speculative decoding
│   └── aarambh-voice-serve/         # HTTP server, multi-format output
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
        ├── align.rs
        ├── learn.rs
        ├── quantise.rs
        ├── eval.rs
        └── serve.rs
```

23 library crates + 1 CLI binary = 24 crates total.

---

## Model Scales

| Scale | d_model | Layers | Heads | KV Heads | Approx Params (transformer core) | Target Use |
|---|---:|---:|---:|---:|---:|---|
| Tiny   | 256  | 6  | 8  | 2 | ~10M  | i3 smoke tests and unit tests |
| Small  | 512  | 12 | 8  | 4 | ~55M  | Kaggle T4 first real checkpoint |
| Medium | 768  | 18 | 12 | 4 | ~170M | Production-quality voice target |
| Large  | 1024 | 24 | 16 | 4 | ~450M | Best-quality research target |

Every subsystem uses the same scale pattern with `AudioDomain::Speech`, `AudioDomain::Music`, or `AudioDomain::Singing`. See `ARCHITECTURE_VOICE_STUDIO_PART1.md` §5 for full detail, and §27 (Part 2) for memory estimates per scale.

---

## Quick Start

### Prerequisites

- Rust stable, 1.80 or later
- Git
- Linux recommended for development
- No GPU required for Phase 0 and CPU smoke tests
- Kaggle GPU or CUDA-capable machine recommended for codec, music, singing, and alignment training

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

28 phases total. Full detail, tasks, and tests for each phase are in `ROADMAP_VOICE_STUDIO_PART1.md` (Phases 0–13) and `ROADMAP_VOICE_STUDIO_PART2.md` (Phases 14–27).

| Phase | Goal | Hardware |
|---:|---|---|
| 0 | Workspace + core types | i3 |
| 1 | Neural audio codec (12.5Hz, transformer bottleneck) | i3 + Kaggle |
| 2 | Text prep — G2P + normalisation | i3 |
| 3 | Data pipeline + auto-labelling | i3 |
| 4 | NN primitives + conditioning injection | i3 |
| 5 | CPU SIMD kernels + CUDA prep | i3 + Kaggle prep |
| 6 | TTS baseline — Tiny trains | i3 + Kaggle |
| 7 | Inference engine + CLI | i3 |
| 8 | Voice cloning | Kaggle |
| 9 | Text-described voice design | i3 + Kaggle |
| 10 | Emotion control | i3 + Kaggle |
| 11 | Music understanding | i3 + Kaggle |
| 12 | Background music generation | Kaggle |
| 13 | Singing synthesis Stage A (autoregressive) | Kaggle |
| 14 | Singing synthesis Stage B (diffusion refinement) | Kaggle |
| 15 | Singing + music mixing | i3 + Kaggle |
| 16 | Cloning + emotion for singing | Kaggle |
| 17 | Structure planner + song composer | i3 + Kaggle |
| 18 | Full control layer | i3 |
| 19 | Safety and watermarking | i3 |
| 20 | Quantisation stack | i3 + Kaggle |
| 21 | Fine-tuning refinement (LoRA/QLoRA/DoRA) | Kaggle |
| 22 | Alignment — GRPO + DPO | Kaggle |
| 23 | Self-learning | i3 |
| 24 | Evaluation harness + baseline comparison | i3 + Kaggle |
| 25 | GPU scale-up + speculative decoding | Kaggle |
| 26 | Inference server + audio output formats | i3 |
| 27 | Production release v1.0 | all |

---

## Safety Policy Summary

Voice generation, singing-voice cloning, and self-learning from user-submitted samples can all be misused. This project therefore treats safety as a core engineering requirement.

Planned safety rules:

- non-preset voice cloning requires a consent token
- generated audio is watermarked
- self-learning updates require the same consent gating as any other cloning-adjacent path
- reference audio is never stored in audit logs, only hashes
- text and lyrics pass through guardrails
- generated content must not impersonate people without permission
- cloned voice examples, speaker packs, and adapters are not bundled by default

See `SECURITY.md` for vulnerability reporting and security scope.

---

## Source Release Policy

This project follows the same source-first discipline as `aarambh-studio`:

- Build from repository source.
- Crates are not published to crates.io until stable.
- `publish = false` should be used during early phases.
- No pretrained checkpoints are included.
- No model weights, voice packs, adapters, cloned voices, self-learned adapter banks, or generated datasets are attached to releases.
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
- WAV/FLAC/Opus read/write utilities
- safe DSP helpers
- CLI help text

Avoid opening large modelling, alignment, or self-learning PRs without an issue first.

---

## License

Apache License 2.0. See [LICENSE](./LICENSE).

---

## Author

Created by **[Darshan Vichhi](https://github.com/aarambh-darshan)** under **[AarambhDevHub](https://github.com/AarambhDevHub)**.