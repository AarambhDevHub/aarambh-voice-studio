# ROADMAP_VOICE_STUDIO.md — aarambh-voice-studio

> Step-by-step build plan. Every phase ends with working, testable code.
> Start Phase 0 today on your i3. No GPU required until Phase 1 (codec
> training benefits from it, but Tiny-scale smoke tests still run on CPU).
> This is a source/engineering release, same policy as `aarambh-ai`: no
> pretrained checkpoints, voice packs, or adapters are released as part of
> this roadmap.

---

## How to Read This Roadmap

Each phase has:
- **Goal** — exactly what you will have when this phase is done
- **Tasks** — the checklist to follow, in order, grouped by crate
- **Tests** — what you write to prove it works
- **Milestone** — how you know you are done, with the git tag to cut

Work top to bottom. Do not skip phases — each phase depends on the ones
before it. Phase 11 (background music generation) is the single heaviest
phase in the whole roadmap; everything before it is deliberately sequenced
to de-risk it (understanding before generation, codec proven on speech
first).

---

## Phase Map (Quick Reference)

```
Phase 0  →  Workspace + core types                    (1–2 days)    [i3]
Phase 1  →  Neural audio codec                         (10–14 days)  [i3 + Kaggle]
Phase 2  →  Data pipeline + auto-labelling             (5–7 days)    [i3]
Phase 3  →  NN primitives + conditioning injection     (5–7 days)    [i3]
Phase 4  →  CPU SIMD kernels + CUDA prep               (5–7 days)    [i3 + Kaggle prep]
Phase 5  →  TTS baseline — Tiny trains!                (10–14 days)  [i3 + Kaggle]
Phase 6  →  Inference engine + CLI                     (5–7 days)    [i3]
Phase 7  →  Voice cloning (zero-shot)                  (7–10 days)   [Kaggle]
Phase 8  →  Voice design (text-described voice)        (5–7 days)    [i3 + Kaggle]
Phase 9  →  Emotion control system                     (7–10 days)   [i3 + Kaggle]
Phase 10 →  Music understanding                        (7–10 days)   [i3 + Kaggle]
Phase 11 →  Background music generation                (14–21 days)  [Kaggle] ⚠ heaviest
Phase 12 →  Singing synthesis (a cappella)              (10–14 days)  [Kaggle]
Phase 13 →  Singing + music mixing                     (7–10 days)   [i3 + Kaggle]
Phase 14 →  Cloning + emotion extended to singing      (7–10 days)   [Kaggle]
Phase 15 →  Lyrics-to-song composer                    (7–10 days)   [i3 + Kaggle]
Phase 16 →  Full control layer                         (5–7 days)    [i3]
Phase 17 →  Safety & watermarking                      (7–10 days)   [i3]
Phase 18 →  Quantisation stack                         (7–10 days)   [i3 + Kaggle]
Phase 19 →  Fine-tuning refinement (LoRA/QLoRA/DoRA)   (7–10 days)   [Kaggle]
Phase 20 →  Evaluation harness                         (7–10 days)   [i3 + Kaggle]
Phase 21 →  GPU scale-up (Small → Large)               (5–7 days)    [Kaggle]
Phase 22 →  Inference server                           (7–10 days)   [i3]
Phase 23 →  Production release v1.0                    (7–10 days)   [all]
```

**Total realistic estimate: 156–216 days (~5.2–7.2 months)** part-time,
consistent with this being a larger scope than `aarambh-ai` v2.0's
99–140 days — audio has more independent subsystems and Phase 11 alone is
heavier than any single phase in the text roadmap.

---

## Why This Order

1. **0–4 first** — workspace, codec, data pipeline, NN primitives, and
   kernels are pure infrastructure. None of them commit to a specific
   engine yet, and the codec (Phase 1) is the single riskiest piece of
   infrastructure in the project — proving it early on speech (the
   best-tooled, most-public-data domain) de-risks reusing it for music and
   singing later.
2. **5–9 (Voice Engine)** comes next because speech has the most mature
   open tooling and public datasets of the three domains — it's the engine
   most likely to work on the first real attempt, and every conditioning
   mechanism built here (speaker embedding, emotion embedding) is reused
   verbatim by the Singing Engine in Phase 14.
3. **10–11 (Music Engine)** — understanding before generation, deliberately.
   The classifier from Phase 10 becomes the auto-labelling tool that makes
   Phase 11's training data affordable, and later becomes an eval metric
   (Phase 20).
4. **12–14 (Singing Engine)** comes after both Voice and Music because
   singing needs melody conditioning (new) plus everything already built
   for cloning and emotion (reused, not rebuilt).
5. **15 (Composer)** is low modelling risk by construction — it only
   orchestrates Phases 5–14, which already exist.
6. **16–17 (Control layer, Safety)** wrap the whole system in one API and
   one consent/watermarking layer before anything ships externally.
7. **18–21 (Quantisation, fine-tuning refinement, eval, GPU scale-up)**
   mirror `aarambh-ai`'s own late-stage phases — refine and measure what
   you already built, don't add new capability surface this late.
8. **22–23 (Server, release)** are last, exactly like `aarambh-ai`'s
   Phase 15/27 discipline: serve a proven system, ship code as source once
   it works, never ship unproven code or bundled weights.

---

## Workspace `Cargo.toml` (write this first, never change it)

```toml
[workspace]
members = [
    "crates/aarambh-voice-core",
    "crates/aarambh-voice-codec",
    "crates/aarambh-voice-data",
    "crates/aarambh-voice-nn",
    "crates/aarambh-voice-kernel",
    "crates/aarambh-voice-model",
    "crates/aarambh-voice-weights",
    "crates/aarambh-voice-train",
    "crates/aarambh-voice-quant",
    "crates/aarambh-voice-finetune",
    "crates/aarambh-voice-speaker",
    "crates/aarambh-voice-emotion",
    "crates/aarambh-voice-music",
    "crates/aarambh-voice-sing",
    "crates/aarambh-voice-mix",
    "crates/aarambh-voice-compose",
    "crates/aarambh-voice-safety",
    "crates/aarambh-voice-eval",
    "crates/aarambh-voice-control",
    "crates/aarambh-voice-inference",
    "crates/aarambh-voice-serve",
    "aarambh-voice-studio",
]
resolver = "2"

[workspace.dependencies]
candle-core         = { version = "0.10" }
candle-nn           = { version = "0.10" }
candle-transformers  = { version = "0.10" }
hound               = "3"
symphonia           = { version = "0.5", features = ["all"] }
rubato              = "0.16"
rustfft             = "6"
apodize             = "1"
tokenizers          = "0.21"
deunicode           = "1"
anyhow              = "1"
thiserror           = "2"
serde               = { version = "1", features = ["derive"] }
serde_json          = "1"
toml                = "0.8"
tokio               = { version = "1", features = ["full"] }
clap                = { version = "4", features = ["derive"] }
tracing             = "0.1"
tracing-subscriber  = "0.3"
safetensors         = "0.8"
rayon               = "1"
cc                  = "1"
which               = "6"
criterion           = "0.5"
sha2                = "0.10"
axum                = "0.8"
```

> **Per-crate Cargo.toml:** when you `cargo new` each crate, add
> `[dependencies]` using `workspace = true`. See ARCHITECTURE_VOICE_STUDIO.md
> §15 for the exact dependency list per crate.

---

## Phase 0 — Workspace + Core Types

**Duration:** 1–2 days | **Hardware:** i3

### Goal
A compilable Cargo workspace where `cargo check --workspace` passes with
zero errors and zero warnings. `aarambh-voice-core` is 100% complete. All
other crates exist as scaffold modules for later phases.

### Tasks

```
[ ] Create directory: aarambh-voice-studio/
[ ] Write root Cargo.toml (copy from above)
[ ] cargo new --lib crates/aarambh-voice-core
[ ] cargo new --lib crates/aarambh-voice-codec
[ ] cargo new --lib crates/aarambh-voice-data
[ ] cargo new --lib crates/aarambh-voice-nn
[ ] cargo new --lib crates/aarambh-voice-kernel
[ ] cargo new --lib crates/aarambh-voice-model
[ ] cargo new --lib crates/aarambh-voice-weights
[ ] cargo new --lib crates/aarambh-voice-train
[ ] cargo new --lib crates/aarambh-voice-quant
[ ] cargo new --lib crates/aarambh-voice-finetune
[ ] cargo new --lib crates/aarambh-voice-speaker
[ ] cargo new --lib crates/aarambh-voice-emotion
[ ] cargo new --lib crates/aarambh-voice-music
[ ] cargo new --lib crates/aarambh-voice-sing
[ ] cargo new --lib crates/aarambh-voice-mix
[ ] cargo new --lib crates/aarambh-voice-compose
[ ] cargo new --lib crates/aarambh-voice-safety
[ ] cargo new --lib crates/aarambh-voice-eval
[ ] cargo new --lib crates/aarambh-voice-control
[ ] cargo new --lib crates/aarambh-voice-inference
[ ] cargo new --lib crates/aarambh-voice-serve
[ ] cargo new --bin aarambh-voice-studio
```

**Write `aarambh-voice-core` completely:**

```
[ ] src/config.rs
      AudioDomain { Speech, Music, Singing }
      ModelConfig { d_model, n_layers, n_heads, n_kv_heads, max_frames,
                    n_codebooks, domain: AudioDomain }
      impl ModelConfig { fn tiny() / small() / medium() / large() -> Self }
[ ] src/request.rs
      NaadRequest, VoiceSpec, EmotionSpec, SingingSpec, MusicSpec, MixSpec
      (full fields defined in Phase 16 — stub structs here, fleshed out later)
[ ] src/error.rs
      AarambhVoiceError via thiserror, one variant per crate's failure mode
[ ] lib.rs re-exports
```

### Tests
```
[ ] ModelConfig::tiny() produces expected d_model/n_layers for each AudioDomain
[ ] Config round-trips through serde_json without loss
```

### Milestone
`cargo check --workspace` is clean. Tag: `v0.1.0-phase0`

---

## Phase 1 — Neural Audio Codec

**Duration:** 10–14 days | **Hardware:** i3 (small-scale) + Kaggle (real training)

### Goal
A working RVQ neural audio codec: encode a waveform to discrete tokens,
decode tokens back to a waveform, with reconstruction quality good enough
to be usable as the fixed tokenizer for every later phase.

### Tasks

**`aarambh-voice-codec`:**
```
[ ] src/encoder.rs
      Conv1D downsampling stack (24kHz → ~75 frames/sec), candle-based
[ ] src/rvq.rs
      ResidualVectorQuantizer { n_codebooks: 8, codebook_size: 1024 }
      quantize() / dequantize(), straight-through estimator for gradients
[ ] src/decoder.rs
      Mirror of the encoder — transposed conv upsampling stack
[ ] src/loss.rs
      Reconstruction (multi-scale STFT) + codebook commitment loss
      Adversarial loss deferred to a later refinement pass — start with
      reconstruction + commitment only, this alone is enough for a usable v1
[ ] src/lib.rs
      AudioCodec::encode(&self, wav: &[f32]) -> TokenGrid
      AudioCodec::decode(&self, tokens: &TokenGrid) -> Vec<f32>
```

**`aarambh-voice-kernel`:**
```
[ ] src/stft.rs — fused STFT/mel-spectrogram via rustfft, used by loss.rs
```

### Dependency Policy
`rustfft` + `apodize` are added here for STFT-based losses. `hound` for WAV
I/O, `symphonia` for decoding compressed input formats during dataset prep.

### Tests
```
[ ] encode() then decode() on a 1-second sine wave reconstructs within a
    fixed spectral-distance tolerance
[ ] TokenGrid shape matches expected [n_codebooks, n_frames] for known input length
[ ] Codec is frozen after training: a checksum test on saved weights ensures
    later phases don't accidentally retrain it
```

### Milestone
Encode/decode round-trip on real speech clips is audibly recognisable (a
qualitative check, backed by the spectral-distance test above). Tag:
`v0.1.0-phase1`

---

## Phase 2 — Data Pipeline + Auto-Labelling

**Duration:** 5–7 days | **Hardware:** i3

### Goal
Automated download, preprocessing, and labelling pipelines for speech,
music, and singing datasets — mirroring `aarambh-ai`'s automated
dataset-infra discipline.

### Tasks

**`aarambh-voice-data`:**
```
[ ] src/speech.rs
      LJSpeech / Common Voice / IndicTTS (Hindi/Sanskrit) loaders
      Text-audio pair validation, silence trimming, resampling via rubato
[ ] src/music.rs
      Public-domain instrumental corpus loader
      Auto-labelling hook (calls aarambh-voice-music's understanding model
      once it exists in Phase 10 — stub a manual-label fallback for now)
[ ] src/singing.rs
      Public singing datasets (lyrics + audio + optional MIDI melody)
[ ] src/chunk.rs
      Fixed-length audio chunking + padding, shared across all three loaders
[ ] src/manifest.rs
      JSONL manifest format: { audio_path, text/lyrics, speaker_id?, tags? }
```

### Tests
```
[ ] Loader produces manifests with valid, existing audio paths
[ ] Chunking produces fixed-length tensors with correct padding masks
[ ] Resampling output matches target sample rate exactly
```

### Milestone
`cargo run -p aarambh-voice-data --bin prepare -- --domain speech` produces
a valid manifest + chunked tensors from a small local sample set. Tag:
`v0.1.0-phase2`

---

## Phase 3 — NN Primitives + Conditioning Injection

**Duration:** 5–7 days | **Hardware:** i3

### Goal
The shared transformer block (ported from `aarambh-ai-nn`'s design) plus
the conditioning-injection mechanism every later engine depends on.

### Tasks

**`aarambh-voice-nn`:**
```
[ ] src/attention.rs — GQA, same math as aarambh-ai-nn, causal mask handling
[ ] src/norm.rs — RMSNorm
[ ] src/rope.rs — rotary position embedding
[ ] src/ffn.rs — SwiGLU feed-forward
[ ] src/block.rs — pre-norm residual transformer block, assembles the above
[ ] src/conditioning.rs
      ConditioningInjector — adds speaker_emb / emotion_emb / style_emb to
      the block's input at every layer, not just the embedding layer
      Supports both additive (fixed-size vector) and cross-attention
      (variable-length, for melody contours in Phase 12) conditioning
```

### Tests
```
[ ] Block output shape matches input shape for all four model scales
[ ] ConditioningInjector with a zero vector reproduces unconditioned output
    exactly (regression test — proves conditioning is truly additive/optional)
[ ] Cross-attention conditioning handles variable-length conditioning sequences
```

### Milestone
`cargo test -p aarambh-voice-nn` passes for all four scales. Tag:
`v0.1.0-phase3`

---

## Phase 4 — CPU SIMD Kernels + CUDA Prep

**Duration:** 5–7 days | **Hardware:** i3 + Kaggle prep

### Goal
AVX2/FMA-accelerated RMSNorm and parallel attention on CPU; CUDA feature
flags scaffolded (not yet implemented — that's Phase 21).

### Tasks

**`aarambh-voice-kernel`:**
```
[ ] src/simd_rmsnorm.rs — AVX2/FMA fused RMSNorm with scalar fallback
[ ] src/parallel_attention.rs — rayon-parallelised attention heads on CPU
[ ] build.rs — cc-based CUDA kernel build prep, feature-gated `cuda`, no-op
    on default CPU builds
```

### Tests
```
[ ] SIMD RMSNorm output matches scalar reference within float tolerance
[ ] Feature detection falls back to scalar path correctly on non-AVX2 CPUs
```

### Milestone
Benchmark harness (via `criterion`) shows measurable speedup over the
scalar baseline on the i3. Tag: `v0.1.0-phase4`

---

## Phase 5 — TTS Baseline — Tiny Trains!

**Duration:** 10–14 days | **Hardware:** i3 (Tiny) + Kaggle (Small+)

### Goal
A working end-to-end TTS model: text → phonemes → transformer over audio
codec tokens → codec decode → waveform. Tiny scale trains successfully on
the i3.

### Tasks

**`aarambh-voice-model`:**
```
[ ] src/tts.rs
      TtsModel { phoneme_embed, transformer (aarambh-voice-nn blocks), lm_head }
      forward() takes phoneme sequence, produces audio token logits
[ ] src/phonemes.rs — grapheme-to-phoneme conversion (English first, Hindi/
    Sanskrit phoneme sets added incrementally)
```

**`aarambh-voice-train`:**
```
[ ] src/pretrain.rs
      Stage 1 domain pretraining loop: cross-entropy over audio codec tokens,
      AdamW, cosine schedule, checkpointing — same shape as aarambh-ai-train
[ ] configs/tts_tiny.toml
```

**`aarambh-voice-weights`:**
```
[ ] src/safetensors_io.rs — save/load model + optimizer state
```

### Tests
```
[ ] One training step reduces loss on a fixed toy batch (sanity gradient check)
[ ] Checkpoint save → load reproduces identical forward-pass output
[ ] Full Tiny run on a small LJSpeech subset drops perplexity over audio
    tokens below a fixed threshold within 5K steps
```

### Milestone
`cargo run -p aarambh-voice-train --bin pretrain -- --config configs/tts_tiny.toml`
completes on the i3 and produces a checkpoint that decodes to recognisable
(if rough) speech. Tag: `v0.1.0-phase5`

---

## Phase 6 — Inference Engine + CLI

**Duration:** 5–7 days | **Hardware:** i3

### Goal
`aarambh-voice-studio speak` works end-to-end from the command line.

### Tasks

**`aarambh-voice-inference`:**
```
[ ] src/kv_cache.rs — fixed-capacity KV cache, preallocated at max_frames
[ ] src/generate.rs — autoregressive audio token generation loop, streaming
    frame-by-frame codec decode (see ARCHITECTURE_VOICE_STUDIO.md §8)
```

**`aarambh-voice-studio` (bin):**
```
[ ] src/cmd/speak.rs — `aarambh-voice-studio speak --text "..." --out out.wav`
[ ] src/main.rs — clap CLI wiring
```

### Tests
```
[ ] CLI produces a valid WAV file of the expected sample rate and duration
[ ] Streaming decode produces identical output to non-streaming decode
    (regression test)
```

### Milestone
`aarambh-voice-studio speak --text "hello from aarambh voice studio" --out hello.wav`
produces a playable WAV file. Tag: `v0.1.0-phase6`

---

## Phase 7 — Voice Cloning (Zero-Shot)

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
Clone a voice from 3–10 seconds of reference audio, no per-speaker
training required.

### Tasks

**`aarambh-voice-speaker`:**
```
[ ] src/encoder.rs
      SpeakerEncoder — conv/transformer network, reference audio → 256-dim
      embedding
[ ] src/clone.rs
      clone_from_reference(wav: &[f32]) -> SpeakerEmbedding
[ ] src/lib.rs — wires SpeakerEmbedding into aarambh-voice-nn's
    ConditioningInjector (Phase 3)
```

**`aarambh-voice-train`:**
```
[ ] src/speaker_pretrain.rs
      Trains the speaker encoder on a multi-speaker corpus with a
      speaker-classification or contrastive (GE2E-style) loss
```

### Tests
```
[ ] Embeddings from two clips of the same speaker are closer (cosine
    similarity) than embeddings from different speakers, on held-out data
[ ] Cloned-voice generation produces higher speaker similarity (§14 of
    ARCHITECTURE_VOICE_STUDIO.md's eval metric) than an unconditioned bake
```

### Milestone
`aarambh-voice-studio clone --reference sample.wav --text "..." --out cloned.wav`
produces speech recognisably similar to the reference speaker. Tag:
`v0.1.0-phase7`

---

## Phase 8 — Voice Design (Text-Described Voice)

**Duration:** 5–7 days | **Hardware:** i3 + Kaggle

### Goal
Generate a voice from a text description alone, no reference audio.

### Tasks

**`aarambh-voice-speaker`:**
```
[ ] src/design.rs
      TextToVoiceEncoder — small text encoder, description -> point in the
      same 256-dim speaker-embedding space as Phase 7
[ ] src/caption.rs
      Captioning tool that generates (description, embedding) training
      pairs from existing labelled speaker embeddings (age/gender/tone
      metadata -> template descriptions), used to build the training set
```

### Tests
```
[ ] Same description produces embeddings within a fixed distance across runs
    (determinism check, modulo any intentional sampling temperature)
[ ] Descriptions with opposing attributes (e.g. "deep male" vs "light
    female") map to well-separated regions of embedding space
```

### Milestone
`aarambh-voice-studio design --description "warm, calm, mid-30s female" --text "..." --out designed.wav`
produces speech matching the described voice qualitatively. Tag:
`v0.1.0-phase8`

---

## Phase 9 — Emotion Control System

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
Discrete emotion tags and continuous per-emotion intensity control over
generated speech.

### Tasks

**`aarambh-voice-emotion`:**
```
[ ] src/space.rs
      EmotionSpace — one learned embedding per discrete emotion (joy,
      sadness, anger, fear, calm, excitement, + others), plus a neutral base
[ ] src/control.rs
      blend(intensities: HashMap<Emotion, f32>) -> EmotionEmbedding
      Implements: base_neutral + intensity * (target - base_neutral),
      weighted-summed across multiple simultaneous emotions
[ ] src/lib.rs — wires EmotionEmbedding into ConditioningInjector alongside
    SpeakerEmbedding (both additive at every layer)
```

**`aarambh-voice-train`:**
```
[ ] src/emotion_finetune.rs
      Fine-tunes the emotion embeddings on an emotional-speech dataset with
      discrete emotion labels
```

### Tests
```
[ ] intensity = 0.0 reproduces neutral output exactly (regression test)
[ ] intensity = 1.0 vs 0.5 produces measurably different (not identical)
    output via the emotion-classification eval metric (§14)
[ ] Blending two emotions produces output classified as a mix, not purely
    either one
```

### Milestone
`aarambh-voice-studio speak --text "..." --emotion joy:0.8,calm:0.2 --out out.wav`
produces speech an emotion classifier scores as predominantly joyful. Tag:
`v0.1.0-phase9`

---

## Phase 10 — Music Understanding

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
An audio encoder + classification heads predicting genre, tempo, key,
mood, instrumentation, and energy from an input clip — used standalone and
as an auto-labelling tool for Phase 11's training data.

### Tasks

**`aarambh-voice-music`:**
```
[ ] src/understand/encoder.rs — audio encoder (reuses codec's encoder
    stack as a frozen feature extractor, adds trainable classification heads)
[ ] src/understand/heads.rs — GenreHead, TempoHead (regression, BPM),
    KeyHead, MoodHead, InstrumentHead (multi-label), EnergyHead
[ ] src/understand/lib.rs — MusicUnderstanding::analyze(wav) -> MusicTags
```

**`aarambh-voice-data`:**
```
[ ] Wire MusicUnderstanding::analyze() into music.rs's auto-labelling hook
    from Phase 2, replacing the manual-label fallback
```

### Tests
```
[ ] Tempo head predicts BPM within a fixed tolerance on a labelled test set
[ ] Genre/mood classification accuracy exceeds a fixed baseline on held-out data
[ ] Auto-labelling pipeline produces manifests with populated tag fields
```

### Milestone
`aarambh-voice-studio music analyze --in clip.wav` prints genre/tempo/key/
mood/instrumentation. Tag: `v0.1.0-phase10`

---

## Phase 11 — Background Music Generation ⚠ heaviest phase

**Duration:** 14–21 days | **Hardware:** Kaggle (mandatory — not realistic on i3 alone)

### Goal
Text prompt → instrumental track. The single most compute-intensive phase
in the roadmap — budget the most calendar time here.

### Tasks

**`aarambh-voice-music`:**
```
[ ] src/generate/text_encoder.rs — prompt text -> style embedding
[ ] src/generate/model.rs
      MusicGenModel — transformer over audio codec tokens conditioned on
      style embedding (reuses aarambh-voice-nn's ConditioningInjector)
[ ] src/generate/lib.rs — MusicGenerator::generate(prompt, duration) -> wav
```

**`aarambh-voice-train`:**
```
[ ] src/music_pretrain.rs
      Stage 1 pretraining on the auto-labelled corpus from Phase 10.
      Dependency Policy: prefer LoRA/QLoRA adaptation of an existing open
      music-codec-LM backbone over from-scratch pretraining as the default
      path — from-scratch is a stretch goal only, per
      ARCHITECTURE_VOICE_STUDIO.md §2/§18
```

### Tests
```
[ ] Generated clips score above a fixed threshold on the music-tag-agreement
    eval metric (§14): analyze(generated) tags should match the prompt
[ ] Generation is reproducible given a fixed seed
[ ] Duration parameter produces output within a fixed tolerance of requested length
```

### Milestone
`aarambh-voice-studio music generate --prompt "lo-fi hip-hop, rainy, 80 bpm" --duration 30 --out beat.wav`
produces a recognisable instrumental matching the prompt. Tag:
`v0.1.0-phase11`

---

## Phase 12 — Singing Synthesis (A Cappella)

**Duration:** 10–14 days | **Hardware:** Kaggle

### Goal
Lyrics + melody (pitch contour) + per-syllable duration → sung vocals, no
backing track yet.

### Tasks

**`aarambh-voice-sing`:**
```
[ ] src/melody.rs — MelodyContour { pitch_hz: Vec<f32>, note_durations: Vec<f32> },
    parses simple MIDI-style pitch/duration input
[ ] src/align.rs — aligns lyrics phonemes to melody notes (syllable-to-note mapping)
[ ] src/model.rs
      SingingModel — transformer over audio codec tokens, conditioned on
      phonemes + melody contour (cross-attention, variable length, per
      Phase 3's ConditioningInjector) + speaker_emb + emotion_emb
[ ] src/lib.rs — SingingSynth::sing(lyrics, melody, voice, emotion) -> wav
```

**`aarambh-voice-train`:**
```
[ ] src/singing_pretrain.rs — Stage 1 pretraining on lyrics+audio+melody
    singing datasets from Phase 2
```

### Tests
```
[ ] Generated pitch contour matches requested melody within a fixed cents tolerance
[ ] Syllable timing matches requested note durations within a fixed tolerance
[ ] A cappella output passes the ASR round-trip WER check on the lyrics text
```

### Milestone
`aarambh-voice-studio sing --lyrics lyrics.txt --melody melody.json --out sung.wav`
produces recognisable a cappella singing on-pitch and on-lyrics. Tag:
`v0.1.0-phase12`

---

## Phase 13 — Singing + Music Mixing

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
Combine singing output (Phase 12) with a backing track (Phase 11) into a
finished mix.

### Tasks

**`aarambh-voice-mix`:**
```
[ ] src/align.rs — tempo/beat-match vocal to backing track (time-stretch
    if needed, via rubato)
[ ] src/gain.rs — independent vocal_gain/music_gain, sidechain ducking so
    vocals sit forward
[ ] src/master.rs — LUFS loudness normalisation + simple limiter
[ ] src/lib.rs — Mixer::combine(vocal_wav, music_wav, MixSpec) -> wav
```

### Tests
```
[ ] Combined output duration matches the longer of the two inputs
[ ] Measured LUFS of output matches the target within a fixed tolerance
[ ] Vocal remains intelligible (ASR round-trip WER) after mixing at default gains
```

### Milestone
`aarambh-voice-studio sing --lyrics lyrics.txt --melody melody.json --background beat.wav --out song.wav`
produces a mixed, listenable track. Tag: `v0.1.0-phase13`

---

## Phase 14 — Cloning + Emotion Extended to Singing

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
The same zero-shot cloning (Phase 7) and emotion control (Phase 9)
mechanisms, now conditioning the Singing Engine.

### Tasks

**`aarambh-voice-sing`:**
```
[ ] Wire SpeakerEmbedding (aarambh-voice-speaker) and EmotionEmbedding
    (aarambh-voice-emotion) into SingingModel's ConditioningInjector calls
    — no new conditioning mechanism, pure integration
[ ] src/finetune.rs — optional DoRA fine-tune path for a specific singer's
    technique (vibrato style, breath patterns), via aarambh-voice-finetune
```

### Tests
```
[ ] Cloned singing voice scores similarly on speaker-similarity metric as
    cloned speech from Phase 7
[ ] Emotion-conditioned singing (e.g. sad:0.9) is distinguishable from
    neutral singing via the emotion-classification eval metric
```

### Milestone
`aarambh-voice-studio sing --lyrics lyrics.txt --melody melody.json --reference singer.wav --emotion sadness:0.7 --out out.wav`
produces cloned, emotionally-conditioned singing. Tag: `v0.1.0-phase14`

---

## Phase 15 — Lyrics-to-Song Composer

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
End-to-end: lyrics + style prompt → finished song, no manual melody
required.

### Tasks

**`aarambh-voice-compose`:**
```
[ ] src/melody_gen.rs
      Small sequence model: lyrics syllable structure + style prompt ->
      MelodyContour (pitch/duration pairs) — only needed if the caller
      doesn't supply one
[ ] src/pipeline.rs
      compose(lyrics, style_prompt, voice, emotion) ->
        1. melody_gen (if no melody given)
        2. aarambh-voice-music::generate() for backing track
        3. aarambh-voice-sing::sing() conditioned on the melody
        4. aarambh-voice-mix::combine()
[ ] src/lib.rs — SongComposer::compose(...) -> wav
```

### Tests
```
[ ] Auto-generated melody stays within a singable pitch range for the
    requested voice
[ ] Full pipeline runs end-to-end without manual intervention on a test
    lyrics file
[ ] Output passes the same mixing/ASR checks as Phase 13's manual-melody path
```

### Milestone
`aarambh-voice-studio compose --lyrics song.txt --style "upbeat pop, 120 bpm" --voice designed:"bright female" --out finished_song.wav`
produces a complete song from lyrics alone. Tag: `v0.1.0-phase15`

---

## Phase 16 — Full Control Layer

**Duration:** 5–7 days | **Hardware:** i3

### Goal
The unified `NaadRequest` struct (ARCHITECTURE_VOICE_STUDIO.md §7) is fully
implemented and every existing capability is reachable through it.

### Tasks

**`aarambh-voice-control`:**
```
[ ] src/request.rs — full NaadRequest / NaadResponse structs, all fields
    from ARCHITECTURE_VOICE_STUDIO.md §7 implemented (not stubs)
[ ] src/resolve.rs
      resolve(request: NaadRequest) -> NaadResponse
      Validates the request (e.g. cloning requires consent_token — enforced
      here, not just documented), dispatches to the correct combination of
      aarambh-voice-speaker / -emotion / -music / -sing / -mix / -compose
[ ] src/validate.rs — field-level validation, clear error messages for
    invalid combinations (e.g. singing spec without lyrics content)
```

**`aarambh-voice-studio` (bin):**
```
[ ] Refactor all existing subcommands (speak/clone/design/music/sing/
    compose) to build a NaadRequest and call aarambh-voice-control::resolve()
    — single code path for every entry point from here on
```

### Tests
```
[ ] Every existing CLI subcommand still produces identical output after
    the refactor (regression test against Phase 5–15 fixtures)
[ ] Invalid requests (e.g. cloning with no consent_token) are rejected with
    a clear error, not a panic
```

### Milestone
Every capability built in Phases 5–15 is reachable through one
`NaadRequest` struct with no feature regressions. Tag: `v0.1.0-phase16`

---

## Phase 17 — Safety & Watermarking

**Duration:** 7–10 days | **Hardware:** i3

### Goal
Consent gating on cloning, inaudible watermarking on all generated audio,
and text/lyrics content guardrails reused from `aarambh-ai-safety`.

### Tasks

**`aarambh-voice-safety`:**
```
[ ] src/consent.rs
      ConsentToken { statement_hash, method: Recorded | Typed | DeveloperAttested }
      Enforced by aarambh-voice-control::validate() from Phase 16 — any
      VoiceSpec::Cloned request without a token is rejected
[ ] src/watermark.rs
      embed_watermark(wav) -> wav — inaudible signal embedded post-generation
      detect_watermark(wav) -> bool — companion detector
[ ] src/guardrails.rs
      Ports aarambh-ai-safety's input/output guardrail pattern (PII
      detection, disallowed-content filtering) applied to text/lyrics input
[ ] src/audit.rs — logs a hash of reference audio + consent token per
    cloning request, never the raw audio
```

### Tests
```
[ ] Cloning request without a consent_token is rejected end-to-end through
    aarambh-voice-control
[ ] detect_watermark() correctly identifies watermarked vs. unwatermarked audio
[ ] Watermark embedding does not measurably degrade the eval metrics from §14
    (spot-check speaker similarity / WER before and after)
[ ] Guardrails catch disallowed lyrics/text content on a fixed test set
```

### Milestone
No audio leaves `aarambh-voice-control::resolve()` unwatermarked, and no
cloning request without consent succeeds. Tag: `v0.1.0-phase17`

---

## Phase 18 — Quantisation Stack

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
INT8 / INT4 / GGUF-style quantisation of the transformer core across all
engines, for low-latency CPU deployment.

### Tasks

**`aarambh-voice-quant`:**
```
[ ] src/int8.rs — post-training INT8 quantisation
[ ] src/int4.rs — GPTQ/AWQ-style INT4 quantisation
[ ] src/gguf.rs — GGUF-style packed export format
[ ] src/qat.rs — optional quantisation-aware training pass
```

### Tests
```
[ ] Quantised model output stays within a fixed quality-metric tolerance
    (WER / speaker similarity) of the F32 baseline, per engine
[ ] Quantised inference matches the memory estimates in
    ARCHITECTURE_VOICE_STUDIO.md §17
```

### Milestone
`aarambh-voice-studio quantise --engine tts --scale medium --format int4`
produces a checkpoint that runs on the i3 within the documented memory
budget. Tag: `v0.1.0-phase18`

---

## Phase 19 — Fine-Tuning Refinement (LoRA / QLoRA / DoRA)

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
A unified fine-tuning path across all engines, formalising the per-speaker,
per-genre, and per-singing-style adapters used ad hoc in earlier phases.

### Tasks

**`aarambh-voice-finetune`:**
```
[ ] src/lora.rs — LoRA adapter injection into aarambh-voice-nn blocks
[ ] src/qlora.rs — QLoRA (quantised base + LoRA adapters)
[ ] src/dora.rs — weight-decomposed DoRA adapters
[ ] src/recipes.rs
      Named recipes: speaker_adapt(), genre_adapt(), singing_style_adapt(),
      language_accent_adapt() — thin wrappers picking sensible target
      modules and rank per use case (ARCHITECTURE_VOICE_STUDIO.md §12)
```

### Tests
```
[ ] LoRA fine-tune on a single speaker improves speaker-similarity metric
    over the base (unadapted) model
[ ] QLoRA fine-tune fits within Small-scale memory budget on Kaggle T4
[ ] DoRA singing-style adapter is distinguishable from base singing style
    via a style-classification spot-check
```

### Milestone
`aarambh-voice-studio finetune --recipe speaker_adapt --data speaker_clips/ --out adapter.safetensors`
produces a usable adapter. Tag: `v0.1.0-phase19`

---

## Phase 20 — Evaluation Harness

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
The full evaluation harness from ARCHITECTURE_VOICE_STUDIO.md §14,
producing a single scorecard across all engines.

### Tasks

**`aarambh-voice-eval`:**
```
[ ] src/asr_roundtrip.rs — WER via ASR round-trip for TTS/singing intelligibility
[ ] src/speaker_sim.rs — cosine similarity between reference and generated
    speaker embeddings
[ ] src/emotion_acc.rs — emotion-classification accuracy on generated output
[ ] src/music_tags.rs — tag agreement between prompt and analyze() output
[ ] src/mos_proxy.rs — learned naturalness predictor
[ ] src/report.rs — Scorecard, to_markdown(), to_json()
```

### Tests
```
[ ] Each metric produces stable, reproducible scores on a fixed fixture set
[ ] Scorecard aggregates all metrics into one report without missing fields
    for any engine
```

### Milestone
`aarambh-voice-studio eval --all` produces a full scorecard across Voice,
Music, and Singing engines. Tag: `v0.1.0-phase20`

---

## Phase 21 — GPU Scale-Up (Small → Large)

**Duration:** 5–7 days | **Hardware:** Kaggle

### Goal
Train Small/Medium/Large scale checkpoints for every engine on Kaggle GPUs,
and implement the CUDA kernels scaffolded in Phase 4.

### Tasks

**`aarambh-voice-kernel`:**
```
[ ] src/cuda/flash_attention.rs — CUDA Flash-Attention-v2-style kernel,
    feature-gated `cuda`
[ ] src/cuda/fused_rmsnorm.rs / fused_rope.rs / fused_swiglu.rs
```

**`aarambh-voice-train`:**
```
[ ] configs/{engine}_{small,medium,large}.toml for every engine
[ ] Throughput benchmark logging (tok/s) per config
```

### Tests
```
[ ] CUDA kernel output matches CPU reference within float tolerance
[ ] Training throughput meets or exceeds the targets in
    ARCHITECTURE_VOICE_STUDIO.md §18's Kaggle GPU table
```

### Milestone
Small and Medium checkpoints exist for every engine, trained end-to-end on
Kaggle free-tier GPUs. Tag: `v0.1.0-phase21`

---

## Phase 22 — Inference Server

**Duration:** 7–10 days | **Hardware:** i3

### Goal
An HTTP server exposing `NaadRequest`/`NaadResponse` over the network, with
chunked streaming audio responses.

### Tasks

**`aarambh-voice-serve`:**
```
[ ] src/server.rs — axum routes, one endpoint per capability plus a
    unified `/generate` endpoint accepting a full NaadRequest
[ ] src/batching.rs — continuous batching across concurrent requests
[ ] src/streaming.rs — chunked-transfer streaming of decoded audio frames,
    reusing aarambh-voice-inference's frame-by-frame decode from Phase 6
[ ] src/session.rs — per-request KV cache + safety-layer pass-through
```

**`aarambh-voice-studio` (bin):**
```
[ ] src/cmd/serve.rs — `aarambh-voice-studio serve --port 8080`
```

### Tests
```
[ ] `/generate` with a TTS-only request returns a valid streamed WAV
[ ] Concurrent requests are batched without cross-request state leakage
[ ] Server rejects unwatermarkable / non-consented requests identically to
    the CLI path (safety layer is not bypassed by the server)
```

### Milestone
`aarambh-voice-studio serve --port 8080` accepts HTTP requests and streams
audio back for every engine. Tag: `v0.1.0-phase22`

---

## Phase 23 — Production Release v1.0

**Duration:** 7–10 days | **Hardware:** all

### Goal
A tagged, documented, source-only v1.0.0 release — same discipline as
`aarambh-ai`'s Phase 15.

### Tasks

```
[ ] Strict docs pass across all 20 crates (crate-level doc comments,
    README per crate)
[ ] CI workflow: cargo check --workspace, cargo test --workspace,
    cargo clippy --workspace -- -D warnings
[ ] RELEASE.md — confirms: source release, crates use publish = false,
    no pretrained checkpoints/adapters/voice packs bundled
[ ] CHANGELOG_VOICE_STUDIO.md — full phase-by-phase history
[ ] Example Tiny configs and local smoke-test data paths documented for
    every engine
[ ] Tag v1.0.0 release notes summarising all 20 crates and 6 major
    capabilities (TTS, cloning, voice design, emotion control, music
    generation/understanding, singing)
```

### Tests
```
[ ] Full workspace builds and tests pass on a clean checkout with only the
    documented prerequisites installed
[ ] Every CLI subcommand documented in the README runs successfully against
    a Tiny checkpoint on a clean i3-equivalent environment
```

### Milestone
```
git checkout v1.0.0
cargo build --release -p aarambh-voice-studio
target/release/aarambh-voice-studio --version
```
Tag: `v1.0.0`
