# ROADMAP_VOICE_STUDIO.md — Part 1 of 2 — aarambh-voice-studio

> Final v1 step-by-step build plan, same format as `aarambh-ai`'s roadmap:
> every phase ends with working, testable code, a git tag, and a clear
> milestone command. Read alongside ARCHITECTURE_VOICE_STUDIO_PART1/2.md
> and SELF_LEARNING_VOICE_STUDIO.md before starting Phase 0.
>
> This is a source/engineering release: no pretrained checkpoints, voice
> packs, or adapters are released as part of this roadmap.

---

## Phase Map (Quick Reference — Final v1, 28 Phases)

```
Phase 0  →  Workspace + core types                       (1–2 days)    [i3]
Phase 1  →  Neural audio codec (12.5Hz, transformer        (14–18 days) [i3 + Kaggle]
             bottleneck, semantic distillation)
Phase 2  →  Text prep — G2P + normalisation                (4–6 days)   [i3]              [NEW]
Phase 3  →  Data pipeline + auto-labelling                 (5–7 days)   [i3]
Phase 4  →  NN primitives + conditioning injection         (5–7 days)   [i3]
Phase 5  →  CPU SIMD kernels + CUDA prep                   (5–7 days)   [i3 + Kaggle prep]
Phase 6  →  TTS baseline — Tiny trains!                    (10–14 days) [i3 + Kaggle]
Phase 7  →  Inference engine + CLI                         (5–7 days)   [i3]
Phase 8  →  Voice cloning (zero-shot)                       (7–10 days)  [Kaggle]
Phase 9  →  Voice design (text-described voice)             (5–7 days)   [i3 + Kaggle]
Phase 10 →  Emotion control system                          (7–10 days)  [i3 + Kaggle]
Phase 11 →  Music understanding                             (7–10 days)  [i3 + Kaggle]
Phase 12 →  Background music generation                    (14–21 days) [Kaggle] ⚠ heaviest
Phase 13 →  Singing synthesis Stage A (a cappella, AR)      (10–14 days) [Kaggle]

── continues in ROADMAP_VOICE_STUDIO_PART2.md ──

Phase 14 →  Singing synthesis Stage B (diffusion refine)   (7–10 days)  [Kaggle]           [NEW]
Phase 15 →  Singing + music mixing                          (7–10 days)  [i3 + Kaggle]
Phase 16 →  Cloning + emotion extended to singing           (7–10 days)  [Kaggle]
Phase 17 →  Structure planner + song composer               (7–10 days)  [i3 + Kaggle]
Phase 18 →  Full control layer                              (5–7 days)   [i3]
Phase 19 →  Safety & watermarking                           (7–10 days)  [i3]
Phase 20 →  Quantisation stack                               (7–10 days)  [i3 + Kaggle]
Phase 21 →  Fine-tuning refinement (LoRA/QLoRA/DoRA)         (7–10 days)  [Kaggle]
Phase 22 →  Alignment — GRPO + DPO                            (10–14 days) [Kaggle]          [NEW]
Phase 23 →  Self-learning                                     (7–10 days)  [i3]              [NEW]
Phase 24 →  Evaluation harness + baseline comparison         (7–10 days)  [i3 + Kaggle]
Phase 25 →  GPU scale-up + speculative decoding               (7–10 days)  [Kaggle]
Phase 26 →  Inference server + audio output formats           (7–10 days)  [i3]
Phase 27 →  Production release v1.0                           (7–10 days)  [all]
```

**Total realistic estimate: 195–265 days (~6.5–8.8 months)** part-time —
up from the draft's 156–216 days, reflecting the four added phases
(text-prep, singing Stage B, alignment, self-learning) plus the larger
scope of Phase 1 (codec redesign) and Phase 17 (structure planner folded
in). This is the honest number for the version you said you want built
once, completely, with nothing missing.

---

## Why This Order (updated)

1. **0–5 first** — workspace, codec, text-prep, data pipeline, NN
   primitives, kernels: pure infrastructure. The codec (Phase 1) is still
   the single riskiest piece — now with the added risk of the
   transformer-bottleneck + semantic-distillation redesign, which is why
   its duration grew from the draft's 10–14 days to 14–18.
2. **Text-prep is now its own phase (2)**, before data pipeline, because
   the data pipeline's preprocessing step depends on having G2P and
   normalisation already working — sequencing it after codec but before
   data pipeline avoids building data preprocessing around a stub.
3. **6–10 (Voice Engine)** unchanged in spirit from the draft — most
   mature open tooling, most likely to work first, and every conditioning
   mechanism here (speaker embedding, emotion embedding) is reused
   verbatim by the Singing Engine later.
4. **11–12 (Music Engine)** — understanding before generation, unchanged.
5. **13–14 (Singing Engine, now two phases instead of one)** — Stage A
   (autoregressive, matches the rest of the codebase) ships and is
   testable on its own before Stage B (diffusion refinement) is
   attempted. This split exists specifically so a Stage-A-only release is
   still a complete, shippable Singing Engine if Stage B runs into
   trouble — the diffusion refinement head is additive, not a blocker.
6. **15–17 (Mixing, extended cloning/emotion, Composer + structure
   planner)** — the structure planner is folded into the Composer phase
   rather than given its own phase, since it's meaningless without the
   orchestration logic it feeds into.
7. **18–19 (Control layer, Safety)** wrap the system in one API and one
   consent/watermarking layer before anything ships externally — unchanged
   from draft.
8. **20–23 (Quantisation, Fine-tuning, Alignment, Self-learning)** — this
   is the most-changed block vs. the draft. Alignment (22) comes *after*
   fine-tuning refinement (21) deliberately: GRPO/DPO refine general
   quality on top of a model that already fine-tunes well per-speaker;
   doing it in the other order would mean re-aligning after every future
   fine-tune. Self-learning (23) comes last in this block because it's an
   inference-time capability, not a training-time one — it needs the
   eval harness's confidence-gating hooks, which is why Evaluation (24)
   is scaffolded conceptually before Self-learning is fully wired, even
   though the harness's *own* dedicated phase comes right after.
9. **24 (Evaluation + baseline comparison)** now includes the fixed
   external-baseline comparison — added specifically so releases have a
   number to compare against beyond your own prior checkpoint.
10. **25–27 (GPU scale-up + speculative decoding, Server + output formats,
    Release)** — speculative decoding is finalized alongside GPU scale-up
    since it needs both a draft and target checkpoint at real scale to
    benchmark honestly; output formats get their own explicit checklist
    inside the Server phase rather than being assumed.

---

## Workspace `Cargo.toml` (write this first, never change it)

```toml
[workspace]
members = [
    "crates/aarambh-voice-core",
    "crates/aarambh-voice-codec",
    "crates/aarambh-voice-textprep",
    "crates/aarambh-voice-data",
    "crates/aarambh-voice-nn",
    "crates/aarambh-voice-kernel",
    "crates/aarambh-voice-model",
    "crates/aarambh-voice-weights",
    "crates/aarambh-voice-train",
    "crates/aarambh-voice-quant",
    "crates/aarambh-voice-finetune",
    "crates/aarambh-voice-align",
    "crates/aarambh-voice-selflearn",
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
candle-core         = { version = "0.11" }
candle-nn           = { version = "0.11" }
candle-transformers  = { version = "0.11" }
hound               = "3"
symphonia           = { version = "0.5", features = ["all"] }
rubato              = "0.16"
rustfft             = "6"
apodize             = "1"
flacenc             = "0.5"
audiopus             = "0.3"
mp3lame-encoder     = "0.2"
tokenizers          = "0.22"
deunicode           = "1"
anyhow              = "1"
thiserror           = "2"
serde               = { version = "1", features = ["derive"] }
serde_json          = "1"
toml                = "0.9"
tokio               = { version = "1", features = ["full"] }
clap                = { version = "4", features = ["derive"] }
tracing             = "0.1"
tracing-subscriber  = "0.3"
safetensors         = "0.7"
rayon               = "1.7"
cc                  = "1"
which               = "6"
criterion           = "0.8"
sha2                = "0.10"
axum                = "0.8"
```

> **Per-crate Cargo.toml:** when you `cargo new` each crate, add
> `[dependencies]` using `workspace = true`. See
> ARCHITECTURE_VOICE_STUDIO_PART2.md §25 for the exact dependency list
> per crate.

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
[ ] Write root Cargo.toml (copy from above — 23 lib crates + 1 bin)
[ ] cargo new --lib crates/aarambh-voice-core
[ ] cargo new --lib crates/aarambh-voice-codec
[ ] cargo new --lib crates/aarambh-voice-textprep
[ ] cargo new --lib crates/aarambh-voice-data
[ ] cargo new --lib crates/aarambh-voice-nn
[ ] cargo new --lib crates/aarambh-voice-kernel
[ ] cargo new --lib crates/aarambh-voice-model
[ ] cargo new --lib crates/aarambh-voice-weights
[ ] cargo new --lib crates/aarambh-voice-train
[ ] cargo new --lib crates/aarambh-voice-quant
[ ] cargo new --lib crates/aarambh-voice-finetune
[ ] cargo new --lib crates/aarambh-voice-align
[ ] cargo new --lib crates/aarambh-voice-selflearn
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
      NaadRequest, VoiceSpec, EmotionSpec, SingingSpec, MusicSpec, MixSpec,
      AudioOutputFormat, ConsentSpec
      (full fields defined in Phase 18 — stub structs here, fleshed out later)
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
`cargo check --workspace` passes with zero warnings across all 24 crates.
Tag: `v0.1.0-phase0`

---

## Phase 1 — Neural Audio Codec

**Duration:** 14–18 days | **Hardware:** i3 + Kaggle

### Goal
A frozen, benchmarked codec: waveform ⇄ discrete tokens at 12.5 Hz, with a
transformer bottleneck and semantic distillation from a frozen SSL
feature extractor. This is the single riskiest phase — do not proceed to
Phase 2 until the freeze criterion (below) passes.

### Tasks

**`aarambh-voice-codec`:**
```
[ ] src/encoder.rs — strided conv encoder, downsample to 12.5Hz
[ ] src/bottleneck.rs — transformer bottleneck (2-4 layers), encoder + decoder side
[ ] src/rvq.rs — split RVQ: RVQ-1 (2048 codes, semantic) + RVQ-2..8
    (1024 codes each, acoustic residual)
[ ] src/semantic_distill.rs — frozen SSL feature extractor wrapper (run
    once, cache features), cosine-distance distillation loss
[ ] src/discriminator.rs — Multi-Period Discriminator (MPD) + Multi-Scale
    Discriminator (MSD), hinge loss
[ ] src/decoder.rs — transposed conv decoder, upsample back to 24kHz
[ ] src/losses.rs — L_reconstruction (L1 + multi-res STFT),
    L_adversarial, L_feature_matching, L_vq_commitment, L_semantic_distill
[ ] src/lib.rs — Codec::encode(), Codec::decode(), Codec::train_step()
```

### Tests
```
[ ] Codec round-trips a known waveform within a fixed STOI tolerance
[ ] RVQ-1 embedding cosine similarity to frozen SSL teacher exceeds 0.85
    on held-out audio
[ ] Discriminator loss decreases monotonically over the first 10k steps
    on a smoke-test subset (regression guard against a broken adversarial
    setup)
[ ] Frame rate is exactly 12.5Hz for all input lengths tested
```

### Milestone
On held-out LJSpeech-subset audio: STOI ≥ 0.90, ASR-roundtrip WER within
2 points of uncompressed baseline, semantic cosine similarity ≥ 0.85.
Tag `v0.1.0-codec-frozen` — **from this point forward,
`aarambh-voice-codec` is read-only for the rest of the roadmap.**

---

## Phase 2 — Text Prep: G2P + Normalisation *(NEW)*

**Duration:** 4–6 days | **Hardware:** i3

### Goal
A standalone, explicitly-owned text preprocessing crate: grapheme-to-
phoneme conversion and text normalisation (numbers, abbreviations, dates
→ spoken form), with Sanskrit/Hindi/English code-switching handled
explicitly rather than falling back silently to `deunicode` transliteration.

### Tasks

**`aarambh-voice-textprep`:**
```
[ ] src/normalize.rs — numbers → words, abbreviation expansion, date/time
    → spoken form, per-language rule tables (English first, Hindi/Sanskrit
    transliteration-aware)
[ ] src/g2p.rs — grapheme-to-phoneme, dictionary-first with a small
    learned fallback model for out-of-dictionary words
[ ] src/codeswitch.rs — language-boundary detection within a single
    utterance (English/Hindi/Sanskrit mixed text), routes each span to the
    right G2P rules
[ ] src/lib.rs — TextPrep::process(text: &str) -> PhonemeSequence
```

### Tests
```
[ ] Numbers, dates, and common abbreviations normalize correctly against
    a fixed test-case list
[ ] G2P output matches a reference pronunciation dictionary for a held-out
    word list
[ ] Code-switched input (English sentence with an embedded Sanskrit term)
    routes each span through the correct G2P path
```

### Milestone
`aarambh-voice-studio textprep --text "मेरा नाम Darshan है, phase 2 पूरा हो गया"`
produces a correct phoneme sequence spanning both languages. Tag:
`v0.1.0-phase2`

---

## Phase 3 — Data Pipeline + Auto-Labelling

**Duration:** 5–7 days | **Hardware:** i3

### Goal
Dataset loaders and preprocessing for every corpus type used later
(single-speaker TTS, multi-speaker cloning, music, singing), with
auto-labelling scaffolding ready for Phase 11's classifier to plug into.

### Tasks

**`aarambh-voice-data`:**
```
[ ] src/loaders/speech.rs — LJSpeech-style and VCTK-style loaders
[ ] src/loaders/music.rs — FMA/MTG-Jamendo-style loaders
[ ] src/loaders/singing.rs — Opencpop-style loader, MIDI alignment parsing
[ ] src/preprocess.rs — resampling (via rubato), silence trimming, mel/STFT
    caching, calls into aarambh-voice-textprep for any text-paired data
[ ] src/align.rs — Montreal Forced Aligner integration (external tool,
    invoked via subprocess) for ground-truth phoneme durations
[ ] src/autolabel.rs — stub trait `AutoLabeller`, implemented for real in
    Phase 11 once the music-understanding classifier exists
```

### Tests
```
[ ] Each loader produces correctly-shaped batches from a small fixture
    dataset
[ ] Forced-alignment output durations sum to the total utterance length
    within a small tolerance
```

### Milestone
`cargo test -p aarambh-voice-data` passes against fixture data for all
three domains (speech, music, singing). Tag: `v0.1.0-phase3`

---

## Phase 4 — NN Primitives + Conditioning Injection

**Duration:** 5–7 days | **Hardware:** i3

### Goal
The shared transformer core (ported from `aarambh-ai-nn` patterns) with
all conditioning injection points wired: speaker embedding, emotion
embedding, melody/duration — see ARCHITECTURE_VOICE_STUDIO_PART1.md §7.

### Tasks

**`aarambh-voice-nn`:**
```
[ ] src/rmsnorm.rs / rope.rs / gqa.rs / swiglu.rs — ported building blocks
[ ] src/block.rs — TransformerBlock combining the above
[ ] src/conditioning.rs — speaker_embedding broadcast-add (layer 0),
    emotion cross-attention injection points, melody/duration
    cross-attention (used later by Singing Engine)
[ ] src/lib.rs — TransformerCore::forward(tokens, conditioning) -> logits
```

### Tests
```
[ ] TransformerBlock output shape matches input shape for all four
    ModelConfig scales
[ ] Conditioning injection changes the output distribution measurably
    (sanity check: same tokens, different speaker_embedding, produces
    different logits) — a regression guard against conditioning being
    silently a no-op
```

### Milestone
`cargo test -p aarambh-voice-nn` passes for all four scales.
Tag: `v0.1.0-phase4`

---

## Phase 5 — CPU SIMD Kernels + CUDA Prep

**Duration:** 5–7 days | **Hardware:** i3 + Kaggle prep

### Goal
Hand-fused CPU kernels for the transformer's hot paths and the codec's
STFT computation, plus CUDA kernel scaffolding (not yet activated — real
implementation happens in Phase 25).

### Tasks

**`aarambh-voice-kernel`:**
```
[ ] src/cpu/attention.rs — SIMD-optimized attention (feature-gated,
    fallback to naive implementation if SIMD unavailable)
[ ] src/cpu/rmsnorm.rs — fused RMSNorm
[ ] src/cpu/fused_stft.rs — fused STFT/mel kernel, replacing generic
    rustfft call patterns on the codec/music-classifier critical path
[ ] src/cuda/mod.rs — stub modules for flash_attention.rs,
    fused_rmsnorm.rs, fused_rope.rs, fused_swiglu.rs (empty behind
    `cuda` feature, filled in at Phase 25)
```

### Tests
```
[ ] SIMD attention output matches naive reference implementation within
    float tolerance
[ ] Fused STFT output matches rustfft reference within float tolerance
[ ] Benchmark (criterion): fused kernels show measurable speedup over
    naive baseline on the i3
```

### Milestone
`cargo bench -p aarambh-voice-kernel` shows the fused kernels
outperforming naive baselines. Tag: `v0.1.0-phase5`

---

## Phase 6 — TTS Baseline: Tiny Trains!

**Duration:** 10–14 days | **Hardware:** i3 + Kaggle

### Goal
A working Tiny-scale TTS model: text → phonemes (via `-textprep`) →
codec tokens (via frozen `-codec`) → audio, trained end-to-end on the i3.

### Tasks

**`aarambh-voice-model`:**
```
[ ] src/tts.rs — TTSModel struct combining TransformerCore + duration
    predictor head
```

**`aarambh-voice-train`:**
```
[ ] src/tts_loop.rs — training loop implementing L_tts (ARCHITECTURE
    PART1 §7.3): cross-entropy over RVQ tokens + duration L2 loss
[ ] configs/tts_tiny.toml — matches ARCHITECTURE PART1 §7.4 reference
    schedule
```

### Tests
```
[ ] Training loss decreases monotonically over 1000 steps on the
    LJSpeech subset (smoke test)
[ ] Generated audio from a Tiny checkpoint achieves ASR-roundtrip WER
    below a fixed threshold on a held-out sentence set
```

### Milestone
`aarambh-voice-studio speak --text "..." --scale tiny --out output.wav`
produces intelligible speech. Tag: `v0.1.0-phase6`

---

## Phase 7 — Inference Engine + CLI

**Duration:** 5–7 days | **Hardware:** i3

### Goal
`aarambh-voice-inference` with KV caching, and a working CLI binary
wrapping the Phase 6 TTS model. Speculative-decoding scaffolding added
here (draft-model interface defined, not yet benchmarked — real
activation is Phase 25).

### Tasks

**`aarambh-voice-inference`:**
```
[ ] src/kv_cache.rs — per-request KV cache, freed on completion
[ ] src/generate.rs — autoregressive sampling loop (temperature, top-k/p)
[ ] src/speculative.rs — stub trait `DraftModel`, single-model fallback
    path implemented now, dual-model verification implemented in Phase 25
```

**`aarambh-voice-studio` (bin):**
```
[ ] src/cmd/speak.rs — `aarambh-voice-studio speak --text "..." --out out.wav`
[ ] src/main.rs — clap CLI scaffolding for all future subcommands
```

### Tests
```
[ ] KV cache produces identical output to a non-cached forward pass
    (correctness, not just speed)
[ ] CLI `speak` subcommand runs end-to-end on a Tiny checkpoint
```

### Milestone
`aarambh-voice-studio speak --text "Thanks for watching and happy coding!" --out demo.wav`
produces a valid WAV file. Tag: `v0.1.0-phase7`

---

## Phase 8 — Voice Cloning (Zero-Shot)

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
Zero-shot speaker cloning from 3-10 seconds of reference audio, per
ARCHITECTURE_VOICE_STUDIO_PART1.md §8.

### Tasks

**`aarambh-voice-speaker`:**
```
[ ] src/encoder.rs — Conformer-style speaker encoder (~5-10M params),
    outputs 256-dim speaker_embedding
[ ] src/train.rs — GE2E loss + consistency loss (§8.2), consistency
    computed every 4th step on Kaggle T4
[ ] src/clone.rs — Speaker::embed(reference_audio) -> [f32; 256]
```

### Tests
```
[ ] Speaker embeddings for the same speaker (different utterances)
    cluster tighter than embeddings across different speakers (GE2E
    sanity check)
[ ] Cloned output's speaker-similarity score exceeds a fixed threshold
    against the reference embedding
```

### Milestone
`aarambh-voice-studio clone --text "..." --reference ref.wav --out cloned.wav`
produces audio recognizably similar to the reference speaker.
Tag: `v0.1.0-phase8`

---

## Phase 9 — Voice Design (Text-Described Voice)

**Duration:** 5–7 days | **Hardware:** i3 + Kaggle

### Goal
Text-described voice design without reference audio, per ARCHITECTURE
PART1 §9 — a projection MLP mapping text descriptions into the same
speaker-embedding space used by cloning.

### Tasks

**`aarambh-voice-speaker`** (extends Phase 8's crate):
```
[ ] src/design.rs — projection MLP: text embedding -> synthetic
    speaker_embedding
[ ] src/train_design.rs — L_voice_design (§9.2): TTS loss + contrastive
    embedding-alignment loss against real speaker clusters
```

### Tests
```
[ ] A designed voice's embedding lands within the real-speaker embedding
    manifold (not a degenerate outlier point)
[ ] Two distinct text descriptions produce measurably distinct embeddings
```

### Milestone
`aarambh-voice-studio design --text "..." --voice-description "deep, warm, older male voice" --out designed.wav`
produces audio matching the described voice qualitatively.
Tag: `v0.1.0-phase9`

---

## Phase 10 — Emotion Control System

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
Continuous emotion conditioning (8-dim embedding space), per ARCHITECTURE
PART2 §10.

### Tasks

**`aarambh-voice-emotion`:**
```
[ ] src/embedding.rs — 8-dim continuous emotion space (valence, arousal,
    + 6 learned axes)
[ ] src/encoder.rs — EmotionEncoder: label-or-text -> embedding
[ ] src/train.rs — L_emotion (§10.2): TTS loss + auxiliary classification
    + continuity penalty
```

### Tests
```
[ ] Emotion classifier (auxiliary head) achieves above-chance accuracy on
    a held-out labelled set before being trusted as a loss term
[ ] Adjacent-frame embedding continuity penalty measurably smooths
    emotion transitions in an ablation test
```

### Milestone
`aarambh-voice-studio speak --text "..." --emotion "nervous but trying to hide it" --out emotional.wav`
produces audio with audibly different prosody from a neutral bake.
Tag: `v0.1.0-phase10`

---

## Phase 11 — Music Understanding

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
The multi-task music classifier (genre/tempo/key/mood/instrumentation),
built before the generator, doubling as the Phase 12 auto-labelling tool.

### Tasks

**`aarambh-voice-music`:**
```
[ ] src/understand/encoder.rs — Conformer/transformer encoder over mel-
    spectrogram input
[ ] src/understand/heads.rs — genre (softmax), tempo (regression), key
    (softmax), mood (multi-label), instrumentation (multi-label)
[ ] src/understand/train.rs — L_understand (ARCHITECTURE PART2 §11.2)
```

**`aarambh-voice-data`** (extends Phase 3):
```
[ ] src/autolabel.rs — real implementation of `AutoLabeller` using the
    trained classifier above
```

### Tests
```
[ ] Classifier achieves above a fixed accuracy/F1 threshold per head on
    a held-out public-dataset split
[ ] Auto-labelling a small unlabeled corpus produces plausible labels,
    spot-checked manually against a sample
```

### Milestone
`aarambh-voice-studio analyze --input song.wav` prints genre/tempo/
key/mood/instrumentation tags. Tag: `v0.1.0-phase11`

---

## Phase 12 — Background Music Generation ⚠ Heaviest Phase

**Duration:** 14–21 days | **Hardware:** Kaggle

### Goal
Text-to-instrumental generation at Medium scale, using the frozen codec's
music domain and the Phase 11 classifier as both an auto-labelling tool
and a training-time tag-agreement reward signal.

### Tasks

**`aarambh-voice-music`** (extends Phase 11):
```
[ ] src/generate/model.rs — MusicGenModel: TransformerCore conditioned on
    text-style prompt embedding
[ ] src/generate/train.rs — L_music (ARCHITECTURE PART2 §12.2):
    cross-entropy + tag-agreement term (every 500 steps)
[ ] configs/music_medium.toml — matches PART2 §12.4 reference schedule
```

### Tests
```
[ ] Generated music's tag-agreement score (via Phase 11 classifier)
    exceeds a fixed threshold against the prompt's intended style
[ ] Generation is stable (no divergence/NaN) over a full training run
    on the smoke-test subset
```

### Milestone
`aarambh-voice-studio generate-music --prompt "upbeat lo-fi hip hop, 90 BPM" --duration 20 --out music.wav`
produces recognizable, on-prompt instrumental audio. Tag: `v0.1.0-phase12`

---

## Phase 13 — Singing Synthesis Stage A (A Cappella, Autoregressive)

**Duration:** 10–14 days | **Hardware:** Kaggle

### Goal
Lyrics + melody + duration → sung vocals, autoregressive Stage A only
(Stage B diffusion refinement is Phase 14 — this phase must produce a
complete, shippable Singing Engine on its own).

### Tasks

**`aarambh-voice-sing`:**
```
[ ] src/stage_a/model.rs — SingingModel: TransformerCore + melody
    cross-attention + pitch auxiliary head
[ ] src/stage_a/train.rs — L_sing_A (ARCHITECTURE PART2 §13.2): cross-
    entropy + pitch L2 + duration L2
[ ] src/midi.rs — melody/duration input parsing (MIDI-like representation)
```

### Tests
```
[ ] Generated singing's pitch contour (F0) tracks the input melody within
    a fixed tolerance
[ ] ASR-roundtrip WER on sung lyrics is below a fixed threshold
    (intelligibility check specific to singing, harder than speech)
```

### Milestone
`aarambh-voice-studio sing --lyrics "..." --melody melody.mid --out sung.wav`
produces intelligible, pitch-accurate a cappella singing.
Tag: `v0.1.0-phase13`

---

*Continue to ROADMAP_VOICE_STUDIO_PART2.md for Phases 14–27: Singing
Stage B (diffusion refinement), mixing, extended cloning/emotion,
structure planner + composer, control layer, safety, quantisation,
fine-tuning, alignment (GRPO/DPO), self-learning, evaluation, GPU
scale-up, server + output formats, and the v1.0 release.*
