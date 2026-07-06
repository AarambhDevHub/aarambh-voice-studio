# ARCHITECTURE_VOICE_STUDIO.md — Part 1 of 2 — aarambh-voice-studio

> Final v1 architecture. Read this together with Part 2 and
> SELF_LEARNING_VOICE_STUDIO.md before writing any code. Nothing gets added
> after this — this is the complete surface for v1.0.0.

**Companion documents:**
- `ARCHITECTURE_VOICE_STUDIO_PART2.md` — kernels, quantisation, fine-tuning,
  alignment (GRPO/DPO), safety, eval, crate reference, data flow, memory
  estimates, hardware strategy, audio output formats, relationship to
  `aarambh-ai`, out-of-scope.
- `SELF_LEARNING_VOICE_STUDIO.md` — the self-learning subsystem in full,
  mirroring Manas's associative memory + anti-forgetting design.
- `ROADMAP_VOICE_STUDIO_PART1.md` / `PART2.md` — step-by-step build plan.

---

## Table of Contents (Part 1)

1. [Project Overview](#1-project-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Dependency Versions & Toolchain](#3-dependency-versions--toolchain)
4. [Complete Workspace — 24 Crates](#4-complete-workspace--24-crates)
5. [Model Scales](#5-model-scales)
6. [Neural Audio Codec — Training In Detail](#6-neural-audio-codec--training-in-detail)
7. [Shared Transformer Core — Training In Detail](#7-shared-transformer-core--training-in-detail)
8. [Speaker Conditioning (Zero-Shot Cloning) — Training In Detail](#8-speaker-conditioning-zero-shot-cloning--training-in-detail)
9. [Voice Design (Text-Described Voice) — Training In Detail](#9-voice-design-text-described-voice--training-in-detail)

*(Sections 10–20 continue in Part 2, renumbered to flow as one document —
see the master index at the top of Part 2.)*

---

## 1. Project Overview

**aarambh-voice-studio** is a ground-up audio generation and understanding
system written entirely in Rust, on `candle`. Every codec, conditioning
layer, and training loop is implemented from scratch — no bindings to
PyTorch or any Python inference library, no vendored checkpoints.

### What it is (v1, final)

- **Neural audio codec** — waveform ⇄ discrete tokens, low-frame-rate
  (12.5 Hz target) with a transformer bottleneck and semantic distillation,
  so the codec's own tokens are easy for the downstream transformer to
  model — not just a lossy-compression exercise.
- **Voice Engine** — TTS, zero-shot voice cloning, text-described voice
  design, continuous emotion control.
- **Music Engine** — text-to-instrumental generation, and a music
  understanding encoder (genre/tempo/key/mood/instrumentation) used both
  standalone and as an auto-labelling tool.
- **Singing Engine** — lyrics + melody + duration → sung vocals, a cappella
  or mixed, with an optional diffusion refinement head for the final
  naturalness pass (the one place in this architecture where pure
  autoregressive generation is deliberately not the whole story).
- **Song Composer** — orchestrates all three engines, now with an explicit
  **structure planner** (verse/chorus/bridge boundaries) instead of
  treating a song as one flat sequence.
- **Alignment stage** — GRPO + DPO, both driven by reward signals your own
  eval harness already computes. Not bolted on afterward — designed in
  from v1.
- **Self-learning** — online speaker/style adaptation with anti-forgetting,
  full design in `SELF_LEARNING_VOICE_STUDIO.md`. This is what lets the
  system absorb a new voice or a correction without re-running a full
  fine-tune job every time.
- **Full Control Layer** — one typed `NaadRequest`, every knob explicit,
  nothing behind an opaque preset.

### What changed from the draft architecture

| Draft (v0) | Final (v1) | Why |
|---|---|---|
| RVQ codec, no target frame rate stated | Transformer-bottleneck codec, 12.5 Hz target, semantic distillation | Shorter token sequences = your i3 handles longer clips; semantically-distilled tokens are easier for the transformer to model, not just easier to reconstruct from |
| No RL / preference stage | `aarambh-voice-align`: GRPO + DPO, rewards sourced from `-eval` | You already compute WER, speaker-sim, emotion-accuracy, music-tag agreement, MOS-proxy — wiring these as rewards is nearly free once `-eval` exists |
| No post-deployment learning | `aarambh-voice-selflearn`: associative memory + gradient orthogonalization, mirrors Manas | New speaker/style needs an incremental update, not a full LoRA job every time |
| Pure autoregressive everywhere | Optional diffusion refinement head in `-model`, used by Singing Engine | Singing naturalness is where AR-only most visibly falls behind current open research |
| Composer orchestrates flat lyric sequence | `structure.rs` in `-compose` predicts verse/chorus/bridge boundaries first | Full-song coherence, not just per-frame audio quality, is where open-source song generation still visibly lags closed models |
| Eval harness scores itself only | Eval harness includes fixed external baseline comparison | "Good" needs a reference point outside your own checkpoints |
| G2P/text-normalization implicit inside `-data` | Explicit `aarambh-voice-textprep` crate | Pronunciation correctness for Sanskrit/Hindi/English code-switching deserves an owner, not a buried module |
| No output-format spec | §-formats in Part 2: WAV/FLAC/Opus/MP3 explicitly scoped, with licensing notes | "What file comes out of the CLI" needs to be decided once, not discovered mid-Phase-6 |

### The subsystems, one foundation

```
                    ┌────────────────────────────┐
                    │     Full control layer       │  ← NaadRequest
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │   Song composer               │
                    │   (structure planner first)   │
                    └───┬───────────┬───────────┬──┘
                        │           │           │
                 ┌──────┴───┐ ┌─────┴────┐ ┌────┴──────┐
                 │  Voice    │ │  Music   │ │  Singing  │
                 │  engine   │ │  engine  │ │  engine   │
                 └──────┬────┘ └────┬─────┘ └────┬──────┘
                        └───────────┴─────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │   Shared foundation               │
                    │   codec (12.5Hz) + transformer     │
                    │   core + alignment (GRPO/DPO)      │
                    │   + self-learning                  │
                    └─────────────────────────────────────┘
```

---

## 2. Design Philosophy

| Goal | Decision |
|---|---|
| Reuse, don't rebuild | Transformer block (RMSNorm, RoPE, GQA, SwiGLU) ported from `aarambh-ai-nn` patterns |
| Adaptation over from-scratch pretraining | LoRA/QLoRA/DoRA first; from-scratch pretraining is a stretch goal per subsystem |
| One control surface | Every knob is a typed field on `NaadRequest`, never a hidden preset |
| Separate-then-mix over joint generation | Vocals and instrumentals generated independently, then mixed — more controllable, more debuggable |
| Understanding before generation | Music/emotion classifiers built before the matching generator |
| No opaque cloning | Every cloning path carries a consent flag and an inaudible watermark (§13, Part 2) |
| Source-first releases | `publish = false` until proven; no bundled checkpoints |
| i3 + free Kaggle GPU only | No paid compute assumed anywhere |
| **Reward-aligned, not just loss-aligned** *(new)* | Every generative subsystem has a matching eval metric, and that metric becomes a GRPO/DPO reward — quality targets are never "vibes," they're numbers you already compute |
| **Learn after ship, safely** *(new)* | Self-learning updates are confidence-gated against the eval harness before being committed — an update that regresses quality is rejected automatically, not by hand |
| **Structure before content** *(new)* | The Composer decides song shape before generating audio — mirrors "understanding before generation" but applied to macro-structure, not just classification |

---

## 3. Dependency Versions & Toolchain

> Versions below are current as of mid-2026. Pin exact patch versions with
> `cargo add <crate>` at Phase 0 time rather than copying these verbatim —
> Rust crates move fast; treat this table as "which crate and which major/minor
> line," not a frozen lockfile.

```toml
[workspace.dependencies]
# Tensor backend — shared major/minor with aarambh-ai
candle-core         = { version = "0.11" }
candle-nn           = { version = "0.11" }
candle-transformers = { version = "0.11" }

# Audio I/O and DSP (input/decode)
hound               = "3"            # WAV read/write (also our primary WAV encoder)
symphonia           = { version = "0.5", features = ["all"] } # decode mp3/flac/ogg input
rubato              = "0.16"         # resampling
rustfft             = "6"            # STFT/mel spectrogram
apodize             = "1"            # window functions for STFT

# Audio output encoders (NEW — see §-formats, Part 2)
flacenc             = "0.5"          # FLAC lossless encoding
audiopus             = "0.3"          # Opus encode/decode bindings (libopus)
mp3lame-encoder     = "0.2"          # MP3 encoding, feature-gated `mp3` (LGPL/patent notes in Part 2)

# Text / phonemes
tokenizers          = "0.22"
deunicode           = "1"            # transliteration fallback for Sanskrit/Hindi text prep

# Shared with aarambh-ai
anyhow              = "1"
thiserror           = "2"
serde               = { version = "1", features = ["derive"] }
serde_json          = "1"
toml                = "0.9"
tokio               = { version = "1", features = ["full"] }
clap                = { version = "4", features = ["derive"] }
tracing             = "0.1"
tracing-subscriber  = "0.3"
safetensors         = "0.7"          # matches candle 0.11's own safetensors requirement
rayon               = "1.7"
cc                  = "1"
which               = "6"
criterion           = "0.8"
sha2                = "0.10"

# Serving
axum                = "0.8"
```

> **Note on the codec:** the neural audio codec is implemented from scratch
> in `aarambh-voice-codec` using `candle`, following a residual-vector-
> quantisation design with a transformer bottleneck (Mimi-style) and
> semantic distillation from a frozen SSL feature extractor — see §6.

> **Note on output encoders:** `flacenc` and `audiopus` are pure-Rust /
> safe-bindings and stay in the default feature set. `mp3lame-encoder`
> wraps LAME, which carries its own licensing terms — kept behind a
> `mp3` cargo feature, off by default. See Part 2 §-formats for the full
> rationale and per-format use case.

> **Per-crate Cargo.toml:** each crate's `[dependencies]` uses
> `workspace = true`. Exact per-crate list is in Part 2 §15.

---

## 4. Complete Workspace — 24 Crates

Final v1 crate list. New crates vs. the draft are marked **NEW**.

```
aarambh-voice-studio/
├── Cargo.toml                        # workspace root
├── crates/
│   ├── aarambh-voice-core/           # L0 — config, request/response types, errors
│   ├── aarambh-voice-codec/          # L1 — neural audio codec (12.5Hz, transformer bottleneck)
│   ├── aarambh-voice-data/           # L1 — dataset loaders, auto-labelling
│   ├── aarambh-voice-textprep/       # L1 — G2P + text normalisation        [NEW]
│   ├── aarambh-voice-nn/             # L2 — transformer block + conditioning injection
│   ├── aarambh-voice-kernel/         # L2 — CPU SIMD kernels, CUDA prep, fused STFT
│   ├── aarambh-voice-model/          # L3 — per-engine models + diffusion refinement head
│   ├── aarambh-voice-weights/        # L3 — SafeTensors save/load, checkpoint conversion
│   ├── aarambh-voice-train/          # L4 — pretraining / continued-training loops
│   ├── aarambh-voice-quant/          # L4 — INT8 / INT4 / GGUF-style quantisation
│   ├── aarambh-voice-finetune/       # L5 — LoRA / QLoRA / DoRA adapters
│   ├── aarambh-voice-align/          # L5 — GRPO + DPO alignment                [NEW]
│   ├── aarambh-voice-selflearn/      # L5 — online self-learning (see dedicated doc) [NEW]
│   ├── aarambh-voice-speaker/        # L5 — zero-shot cloning, voice design
│   ├── aarambh-voice-emotion/        # L5 — emotion embedding space
│   ├── aarambh-voice-music/          # L6 — music understanding + generation
│   ├── aarambh-voice-sing/           # L7 — singing synthesis
│   ├── aarambh-voice-mix/            # L7 — vocal + instrumental mixing/mastering
│   ├── aarambh-voice-compose/        # L8 — structure planner + lyrics-to-song orchestrator
│   ├── aarambh-voice-safety/         # L9 — consent gating, watermarking, guardrails
│   ├── aarambh-voice-eval/           # L9 — evaluation harness + baseline comparison
│   ├── aarambh-voice-control/        # L9 — full control API / NaadRequest DSL
│   ├── aarambh-voice-inference/      # L9 — shared inference runtime, KV cache, speculative decoding
│   └── aarambh-voice-serve/          # L10 — HTTP inference server
└── aarambh-voice-studio/             # L11 — bin (CLI)
```

23 library crates + 1 binary = **24 crates total**, up from the draft's 20.
The four additions (`-textprep`, `-align`, `-selflearn`, and the structure
planner folded into `-compose`) are the direct answer to "what's missing."

---

## 5. Model Scales

`ModelConfig` (in `aarambh-voice-core`) defines four scales, shared across
all engines, parameterised by `AudioDomain`:

| Scale | d_model | n_layers | n_heads | n_kv_heads | Params (transformer core) |
|---|---|---|---|---|---|
| Tiny   | 256  | 6  | 8  | 2 | ~10M  |
| Small  | 512  | 12 | 8  | 4 | ~55M  |
| Medium | 768  | 18 | 12 | 4 | ~170M |
| Large  | 1024 | 24 | 16 | 4 | ~450M |

Codec (encoder+decoder, RVQ+transformer bottleneck) is scale-independent
by design — one codec, ~15–25M params, trained once in Phase 1/Stage 0,
frozen for all downstream engine training. This mirrors `aarambh-ai`'s
"tokenizer trained once, frozen thereafter" discipline.

```rust
impl ModelConfig {
    pub fn tiny(domain: AudioDomain) -> Self { /* d_model=256, n_layers=6, ... */ }
    pub fn small(domain: AudioDomain) -> Self { /* d_model=512, n_layers=12, ... */ }
    pub fn medium(domain: AudioDomain) -> Self { /* d_model=768, n_layers=18, ... */ }
    pub fn large(domain: AudioDomain) -> Self { /* d_model=1024, n_layers=24, ... */ }
}
```

---

## 6. Neural Audio Codec — Training In Detail

This is the single riskiest and most-changed piece of infrastructure vs.
the draft. Everything downstream depends on it, so it gets trained first,
frozen, and never touched again once Stage 0 passes its milestone.

### 6.1 Architecture

```
waveform (24kHz)
   │
   ▼
Conv encoder (strided, downsample to 12.5Hz frame rate)
   │
   ▼
Transformer bottleneck (2–4 layers, small d_model)   ← the Mimi-style addition
   │
   ▼
Split RVQ:
   RVQ-1  (codebook size 2048)  ← semantic codes, distilled from frozen SSL features
   RVQ-2..8 (codebook size 1024 each) ← acoustic residual codes
   │
   ▼
Transformer bottleneck (decoder side)
   │
   ▼
Conv decoder (transposed, upsample back to 24kHz)
   │
   ▼
waveform (reconstructed)
```

### 6.2 Loss function (Stage 0 codec training)

Total loss is a weighted sum, computed per-batch:

```
L_codec = λ_recon · L_reconstruction     (L1 waveform + multi-resolution STFT loss)
        + λ_adv   · L_adversarial        (multi-period + multi-scale discriminator, hinge loss)
        + λ_feat  · L_feature_matching   (discriminator intermediate-layer L1)
        + λ_vq    · L_vq_commitment      (standard VQ-VAE commitment loss, per RVQ layer)
        + λ_sem   · L_semantic_distill   (cosine distance: RVQ-1 embedding vs. frozen SSL feature)
```

Typical starting weights: `λ_recon=1.0, λ_adv=1.0, λ_feat=2.0, λ_vq=1.0,
λ_sem=1.0`. The semantic-distillation term is what separates this from a
plain EnCodec/DAC clone — RVQ-1 is trained to predict a frozen
self-supervised feature (e.g. a WavLM/HuBERT-style representation computed
once and cached, not trained jointly), so the resulting tokens carry
linguistic content, not just spectral detail. This is what makes the
downstream transformer's job easier: it's modelling something closer to
"phoneme-ish" tokens on RVQ-1, and fine acoustic detail on the rest.

### 6.3 Discriminators

Two discriminators, adversarial from the start (not added later):
- **Multi-Period Discriminator (MPD)** — reshapes the waveform into 2D
  slices at several prime periods, catches periodicity artifacts.
  Multi-Scale Discriminator (MSD) — operates directly on the waveform at
  several downsampling scales, catches broadband artifacts.
- Trained with the standard GAN alternating schedule: one discriminator
  step per generator step, hinge loss on both sides.

### 6.4 Data

- Stage 0 (Tiny, i3): LJSpeech subset (single-speaker English, ~2 hours),
  used purely for codec smoke-training — proving the reconstruction and
  semantic-distillation losses converge before spending any Kaggle time.
- Stage 0 (Small/Medium, Kaggle T4/P100): multi-speaker multilingual mix
  (LibriTTS-R + a Hindi/Sanskrit-adjacent public corpus for code-switching
  coverage, matching your own linguistic focus) plus a small music/singing
  slice so the same codec generalises beyond pure speech.

### 6.5 Training schedule (reference, Small scale, Kaggle T4)

```toml
batch_size       = 16
segment_seconds  = 1.0          # random crop per training step
learning_rate    = 3e-4         # generator
learning_rate_d  = 3e-4         # discriminator
optimizer        = "AdamW"
betas            = [0.8, 0.99]  # standard for GAN-style codec training
max_steps        = 400000
warmup_steps     = 0            # codec GANs typically skip warmup
lr_schedule      = "exponential_decay"  # gamma ~0.999 per 1000 steps
mixed_precision  = "bf16"
```

### 6.6 Milestone / freeze criterion

Codec is frozen once, on a held-out set:
- STOI ≥ 0.90 at 12.5 Hz / target bitrate band
- WER (ASR-roundtrip through the reconstructed audio) within 2 points of
  the uncompressed-waveform ASR baseline
- Semantic-distillation cosine similarity ≥ 0.85 against the frozen SSL
  teacher on held-out audio

Once frozen: tag `v0.1.0-codec-frozen`, and every subsequent phase treats
`aarambh-voice-codec` as read-only infrastructure — same discipline as
`aarambh-ai`'s tokenizer freeze.

---

## 7. Shared Transformer Core — Training In Detail

### 7.1 Architecture

Ported (not shared as a direct dependency) from `aarambh-ai-nn`:
RMSNorm, RoPE positional encoding, Grouped-Query Attention (GQA), SwiGLU
feed-forward. The port exists because audio conditioning needs injection
points text generation doesn't — speaker embedding, emotion embedding,
melody/duration conditioning are all summed or cross-attended into the
residual stream at specific layers, and coupling that to `aarambh-ai-nn`
directly would tie audio-model changes to text-model releases.

### 7.2 Conditioning injection points

```
input: codec tokens (RVQ-1 primary stream, teacher-forced during training)
   │
   ▼
embedding + RoPE
   │
   ▼
[optional] + speaker_embedding   (broadcast-added, layer 0 only)
   │
   ▼
transformer blocks × n_layers
   │         ▲
   │    cross-attend to emotion_embedding at layers {n_layers//3, 2*n_layers//3}
   │    cross-attend to melody/duration tokens (Singing Engine only) at every layer
   ▼
output head → next-token distribution over RVQ codebook(s)
```

### 7.3 Loss function (Stage 1, TTS baseline pretraining)

```
L_tts = CrossEntropy(predicted_RVQ_tokens, target_RVQ_tokens)
      + λ_dur · L_duration   (predicted phoneme duration vs. forced-aligned ground truth, L2)
```

`λ_dur = 0.1`. Forced alignment for ground-truth durations comes from
Montreal Forced Aligner (MFA) run once over the training corpus as a
preprocessing step in `aarambh-voice-data` — not learned jointly, to keep
Stage 1 training stable and debuggable on the i3.

### 7.4 Training schedule (reference, Tiny scale, i3 CPU)

```toml
batch_size        = 2
grad_accum_steps  = 16          # effective batch size 32
max_audio_seconds = 4
dataset           = "data/ljspeech_subset/"
learning_rate     = 1e-3
optimizer         = "AdamW"
betas             = [0.9, 0.95]
weight_decay      = 0.1
max_steps         = 5000
warmup_steps      = 200
lr_schedule       = "cosine"
device            = "cpu"
dtype             = "f32"
eval_steps        = 500
grad_clip_norm    = 1.0
```

### 7.5 Training schedule (reference, Small scale, Kaggle T4)

```toml
batch_size        = 8
max_audio_seconds = 10
learning_rate     = 3e-4
optimizer         = "AdamW"
betas             = [0.9, 0.95]
weight_decay      = 0.1
max_steps         = 60000
warmup_steps      = 1000
lr_schedule       = "cosine"
device            = "cuda:0"
dtype             = "bf16"
eval_steps        = 2000
grad_clip_norm    = 1.0
```

---

## 8. Speaker Conditioning (Zero-Shot Cloning) — Training In Detail

### 8.1 Architecture

A separate lightweight speaker encoder (Conformer-style, ~5–10M params)
consumes 3–10 seconds of reference audio and produces a fixed-size
`speaker_embedding` (256-dim). This embedding is broadcast-added at layer 0
of the shared transformer core (see §7.2) — the same conditioning path
used at inference for cloning, so training and inference never diverge.

### 8.2 Loss function

```
L_speaker = L_tts (as in §7.3, teacher-forced on the target speaker's audio)
          + λ_gen  · L_generalized_end_to_end   (GE2E loss on speaker embeddings,
                                                  pulls same-speaker embeddings
                                                  together, pushes different-speaker
                                                  embeddings apart)
          + λ_cons · L_consistency              (cosine similarity between the
                                                  speaker embedding extracted from
                                                  the *generated* audio and the
                                                  original reference embedding)
```

`λ_gen = 0.5`, `λ_cons = 0.3`. `L_consistency` requires running the speaker
encoder a second time on generated audio each step — expensive, so it's
computed every 4th step on Tiny/Small (i3/T4) and every step once on
Medium+ (Kaggle P100/A100).

### 8.3 Data

Multi-speaker corpus (VCTK-style: many speakers, few minutes each per
speaker) is the right shape for GE2E — cloning needs breadth of speaker
identity, not depth per speaker. Contrast with §7's TTS-baseline corpus,
which prioritizes depth (LJSpeech: one speaker, many hours) for
intelligibility first.

### 8.4 Consent and watermarking

Every training example and every inference call through this path is
gated by `aarambh-voice-safety` (Part 2 §13) — no cloning path exists in
this codebase that bypasses the consent flag, including during training
data preparation.

---

## 9. Voice Design (Text-Described Voice) — Training In Detail

### 9.1 Architecture

Text description ("deep, warm, older male voice, slight rasp") → frozen
text encoder (reused embedding stack from `aarambh-voice-textprep`) →
small projection MLP → a *synthetic* speaker embedding in the same
256-dim space as §8's real speaker embeddings. No reference audio needed
at inference; the projection MLP is trained to land in the same embedding
space so the rest of the pipeline is unmodified.

### 9.2 Loss function

```
L_voice_design = L_tts (as in §7.3, conditioned on the projected embedding)
               + λ_align · L_embedding_alignment  (L2 distance between the
                                                    projected embedding and
                                                    the nearest real speaker
                                                    embedding cluster whose
                                                    human-written description
                                                    matches the input text —
                                                    contrastive, in-batch negatives)
```

`λ_align = 0.4`. Training data: pair each speaker in the §8 corpus with a
short human-written (or LLM-assisted, human-reviewed) description of their
voice — this becomes the (text, target-embedding-region) supervision.

---

*Continue to Part 2 for sections 10–20: custom kernels, quantisation,
fine-tuning, alignment (GRPO/DPO), safety, evaluation, crate-by-crate
reference, data flow, memory estimates, hardware strategy, audio output
formats, relationship to `aarambh-ai`, and out-of-scope items.*
