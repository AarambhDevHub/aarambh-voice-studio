# ARCHITECTURE_VOICE_STUDIO.md — Part 2 of 2 — aarambh-voice-studio

> Continues directly from Part 1 §9. Read Part 1 first.

## Table of Contents (Part 2)

10. [Emotion Conditioning](#10-emotion-conditioning--training-in-detail)
11. [Music Understanding Encoder](#11-music-understanding-encoder--training-in-detail)
12. [Background Music Generation](#12-background-music-generation--training-in-detail)
13. [Singing Synthesis + Diffusion Refinement](#13-singing-synthesis--diffusion-refinement--training-in-detail)
14. [Vocal + Instrumental Mixing](#14-vocal--instrumental-mixing--training-in-detail)
15. [Song Composer + Structure Planner](#15-song-composer--structure-planner--training-in-detail)
16. [Full Control Layer](#16-full-control-layer-aarambh-voice-control)
17. [KV Cache, Streaming & Speculative Decoding](#17-kv-cache-streaming--speculative-decoding)
18. [Custom Kernels](#18-custom-kernels-aarambh-voice-kernel)
19. [Quantisation](#19-quantisation-aarambh-voice-quant)
20. [Fine-Tuning (LoRA/QLoRA/DoRA)](#20-fine-tuning-aarambh-voice-finetune)
21. [Alignment — GRPO + DPO](#21-alignment--grpo--dpo-aarambh-voice-align)
22. [Self-Learning](#22-self-learning-aarambh-voice-selflearn)
23. [Safety Layer](#23-safety-layer-aarambh-voice-safety)
24. [Evaluation Harness + Baseline Comparison](#24-evaluation-harness--baseline-comparison-aarambh-voice-eval)
25. [Crate-by-Crate Reference — 24 Crates](#25-crate-by-crate-reference--24-crates)
26. [Data Flow Across the Workspace](#26-data-flow-across-the-workspace)
27. [Memory & Compute Estimates](#27-memory--compute-estimates)
28. [Hardware Strategy](#28-hardware-strategy)
29. [Audio Output Formats](#29-audio-output-formats)
30. [Relationship to `aarambh-ai`](#30-relationship-to-aarambh-ai)
31. [What's Explicitly Out of Scope (v1)](#31-whats-explicitly-out-of-scope-v1)

---

## 10. Emotion Conditioning — Training In Detail

### 10.1 Architecture
A continuous emotion embedding space (8-dim: valence, arousal, plus 6
learned axes) rather than discrete categories — this is what enables
"continuous emotion control" instead of picking from a fixed list. An
`EmotionEncoder` (small MLP) maps either (a) a discrete label from labelled
training data, or (b) a text description ("nervous but trying to hide it"),
into this space. Injected via cross-attention at two points in the
transformer stack (§7.2).

### 10.2 Loss function
```
L_emotion = L_tts (§7.3, conditioned on emotion embedding)
          + λ_cls  · L_emotion_classification   (auxiliary classifier head on
                                                  generated audio, cross-entropy
                                                  against the source label —
                                                  reuses the classifier trained
                                                  in §11 for the Music Engine,
                                                  fine-tuned on speech-emotion data)
          + λ_cont · L_continuity                (adjacent-frame emotion embedding
                                                  smoothness penalty, L2 on
                                                  first-difference — prevents
                                                  jarring emotion jumps mid-utterance)
```
`λ_cls = 0.3`, `λ_cont = 0.05`.

### 10.3 Data
RAVDESS / CREMA-D style labelled emotional speech corpora for the discrete
axis supervision; text-description pairing follows the same LLM-assisted
+ human-reviewed process as §9.2's voice descriptions.

---

## 11. Music Understanding Encoder — Training In Detail

### 11.1 Architecture
A Conformer or transformer encoder over mel-spectrogram input, multi-head
classification output: genre (multi-class), tempo (regression, BPM), key
(multi-class, 24 classes), mood (multi-label), instrumentation
(multi-label). This is built **before** the generator (Phase order
unchanged from draft) specifically so it can auto-label training data for
§12.

### 11.2 Loss function
```
L_understand = CrossEntropy(genre) + CrossEntropy(key)
             + L2(tempo_bpm, normalized)
             + BCE(mood_multilabel) + BCE(instrumentation_multilabel)
```
Simple weighted sum, all weights 1.0 initially — this is a standard
multi-task classifier, no exotic loss needed.

### 11.3 Data
Public tagged datasets (FMA, MTG-Jamendo style) for initial training;
once trained, this model auto-labels your own scraped/licensed corpus for
§12, which is the actual point of building it first.

---

## 12. Background Music Generation — Training In Detail

**This is the heaviest phase in the entire roadmap** — flagged in the
roadmap explicitly, and the reason everything before it exists is to
de-risk it.

### 12.1 Architecture
Same shared transformer core (§7), conditioned on a text-style prompt
embedding, generating codec tokens over the *music* domain of the same
codec (frozen, §6) rather than speech. Longer context window than speech
(music structure needs more lookback) — Medium/Large scale only; Tiny is
not expected to produce usable results here (see Hardware Strategy §28).

### 12.2 Loss function
```
L_music = CrossEntropy(predicted_RVQ_tokens, target_RVQ_tokens)
        + λ_tag · L_tag_agreement   (run the §11 classifier on a short
                                     generated excerpt every N steps,
                                     cross-entropy against the *prompt's*
                                     intended genre/mood — this is the
                                     "understanding becomes an eval metric"
                                     loop described in the design philosophy)
```
`λ_tag = 0.2`, applied only every 500 steps (expensive — requires a full
forward pass through a second frozen model).

### 12.3 Data
Auto-labelled via §11 classifier over your own corpus. This is the phase
where auto-labelling isn't optional — hand-labelling music at the scale
needed for Medium/Large training is not realistic for a solo dev, which is
exactly why §11 exists first.

### 12.4 Training schedule (reference, Medium scale, Kaggle P100)
```toml
batch_size        = 4
max_audio_seconds = 20
learning_rate     = 2e-4
optimizer         = "AdamW"
betas             = [0.9, 0.95]
max_steps         = 150000
warmup_steps      = 2000
lr_schedule       = "cosine"
device            = "cuda:0"
dtype             = "bf16"
eval_steps        = 5000
```

---

## 13. Singing Synthesis + Diffusion Refinement — Training In Detail

### 13.1 Architecture — two-stage
```
lyrics + melody (MIDI-like pitch/duration) + speaker/emotion embeddings
   │
   ▼
Stage A: shared transformer core (§7), autoregressive over codec tokens
   │        (same architecture as Voice Engine, + melody cross-attention)
   ▼
Stage B [NEW, optional, feature-gated `diffusion-refine`]:
   a small diffusion/flow-matching decoder that takes Stage A's codec-token
   output as conditioning and refines the final mel/waveform pass —
   this is the one deliberate departure from pure autoregressive
   generation in this architecture, because singing naturalness is
   specifically where AR-only trails current open research.
```

Stage B is intentionally **optional and swappable**: `-sing` works
end-to-end with Stage A alone (matches the rest of the codebase's AR
discipline); Stage B is an additive quality pass you can enable once Stage
A is solid, not a hard dependency for the roadmap's earlier milestones.

### 13.2 Loss function

Stage A:
```
L_sing_A = CrossEntropy(predicted_RVQ_tokens, target_RVQ_tokens)
         + λ_pitch · L_pitch   (predicted F0 contour vs. ground truth, L2,
                                 auxiliary head off the transformer's
                                 hidden state)
         + λ_dur   · L_duration (as in §7.3, but duration here is dictated
                                  by the melody input, not predicted freely)
```
`λ_pitch = 0.2`, `λ_dur = 0.1`.

Stage B (diffusion refinement, if enabled):
```
L_sing_B = E_t [ || ε - ε_θ(x_t, t, cond=StageA_tokens) ||^2 ]
```
Standard denoising-diffusion (or flow-matching, either is acceptable —
flow-matching converges faster in practice and is the recommended default)
objective, conditioned on Stage A's output. Trained *after* Stage A is
frozen for the checkpoint being refined — not jointly, to keep the two
stages debuggable independently.

### 13.3 Data
Public singing corpora (Opencpop-style for reference architecture; for
your own corpus, any licensed a cappella recordings with lyric+MIDI
alignment). Melody/duration ground truth comes from MIDI transcription or
manual annotation on a small seed set, matching the "small seed,
auto-expand via the model itself" pattern already used elsewhere.

---

## 14. Vocal + Instrumental Mixing — Training In Detail

### 14.1 Architecture
Not a generative model in the RVQ-token sense — a lightweight
waveform-domain mixing/mastering network (small U-Net over STFT
magnitude, phase reconstructed via Griffin-Lim or a small neural vocoder
head) that takes independently-generated vocal and instrumental stems and
produces a mixed, mastered output. Kept separate from the generative
engines deliberately (§2 "separate-then-mix").

### 14.2 Loss function
```
L_mix = L1(mixed_output_spectrogram, reference_mix_spectrogram)
      + λ_loud · L_loudness   (LUFS-matching penalty against a broadcast
                                loudness target, e.g. -14 LUFS default)
```
`λ_loud = 0.3`. Reference mixes: any licensed multitrack corpus with
separate stems (e.g. MUSDB18-style) provides (stems → reference mix)
supervision directly.

---

## 15. Song Composer + Structure Planner — Training In Detail

### 15.1 Architecture — structure planner **(NEW)**
```
lyrics (full text, with any user-provided section hints)
   │
   ▼
structure.rs: small sequence-labelling transformer
   │   predicts, per line: {Verse, Chorus, Bridge, Intro, Outro}
   │   + repeat-detection (is this chorus a repeat of an earlier one?)
   ▼
structured plan → dispatched section-by-section to Voice/Singing Engine,
                  reusing the same generated audio for repeated choruses
                  rather than regenerating (cheaper AND more consistent)
   │
   ▼
Music Engine generates a style-consistent instrumental bed for the
whole plan (informed by structure, e.g. bigger arrangement on choruses)
   │
   ▼
Mix (§14) combines everything into the final song
```

### 15.2 Loss function (structure planner)
```
L_structure = CrossEntropy(section_label_per_line)
            + BCE(is_repeat_of_earlier_chorus)
```
Standard sequence-labelling supervision. Training data: publicly available
lyric sheets with section headers already present (a very common format —
"[Verse 1]", "[Chorus]" markup is standard in lyric databases), which
gives free, large-scale supervision without manual annotation.

### 15.3 Orchestration logic
`aarambh-voice-compose` sits above `aarambh-voice-control`; the structure
planner's output is the *first* thing computed for any full-song request,
before any audio generation begins — matches the "understanding before
generation" philosophy applied at the macro/structural level instead of
just the classification level.

---

## 16. Full Control Layer (`aarambh-voice-control`)

`NaadRequest` — every field explicit, no hidden presets:

```rust
pub struct NaadRequest {
    pub text: String,
    pub voice: VoiceSpec,        // { reference_audio: Option<Path>, design_text: Option<String>, speaker_id: Option<String> }
    pub emotion: EmotionSpec,    // { label: Option<EmotionLabel>, description: Option<String>, intensity: f32 }
    pub singing: Option<SingingSpec>,  // { melody: MidiLike, lyrics_alignment: Option<Alignment> }
    pub music: Option<MusicSpec>,      // { style_prompt: String, duration_seconds: f32 }
    pub mix: Option<MixSpec>,          // { target_lufs: f32, vocal_gain_db: f32 }
    pub output_format: AudioOutputFormat,  // see §29
    pub consent: ConsentSpec,     // required whenever `voice.reference_audio` is set — see §23
}
```

---

## 17. KV Cache, Streaming & Speculative Decoding

### 17.1 KV cache
Standard causal KV cache, per-request, held in `aarambh-voice-inference`;
freed on request completion. Same discipline as `aarambh-ai`'s inference
runtime.

### 17.2 Streaming
Chunked streaming (not sub-100ms conversational latency — see §31,
out-of-scope) — audio decoded and returned frame-by-frame as codec tokens
are generated, rather than waiting for the full sequence.

### 17.3 Speculative decoding **(NEW — pulled forward from "v2 later")**
A small draft model (Tiny-scale checkpoint of the same architecture)
proposes several codec tokens ahead; the target model (Small/Medium/Large)
verifies them in a single forward pass, accepting the longest matching
prefix. Same mechanism planned for `aarambh-ai` v2 — implementing it here
in v1 instead of deferring, since CPU inference latency matters more for
audio (users notice audio latency more acutely than text-generation
latency) and the draft-model infrastructure is nearly free once Tiny
checkpoints already exist for every engine.

```
Expected speedup: 1.5–2.5x wall-clock on CPU inference, engine-dependent,
at the cost of holding two checkpoints (draft + target) in memory
simultaneously — budget this into the CPU inference memory table (§27).
```

---

## 18. Custom Kernels (`aarambh-voice-kernel`)

- CPU SIMD kernels for the transformer core's hot paths (attention,
  RMSNorm), matching `aarambh-ai-kernel`'s approach.
- Fused STFT kernel — STFT/mel computation is on the critical path for
  both the codec and the music-understanding encoder, worth hand-fusing
  rather than relying on generic `rustfft` call patterns.
- CUDA kernels (Flash-Attention-v2-style, fused RMSNorm/RoPE/SwiGLU),
  feature-gated `cuda`, implemented in the GPU-scale-up phase — CPU builds
  remain the default.

---

## 19. Quantisation (`aarambh-voice-quant`)

INT8 / INT4 (GPTQ/AWQ-style) / GGUF-style export, plus an optional
quantisation-aware-training (QAT) pass — post-hoc quantisation is the
default path (matches `aarambh-ai` v1 discipline), QAT is available for
whichever checkpoint benefits most once post-hoc quality is measured
against the tolerance budget in the eval harness (§24).

---

## 20. Fine-Tuning (`aarambh-voice-finetune`)

LoRA / QLoRA / DoRA adapters, injected into `aarambh-voice-nn` blocks.
Named recipes (`speaker_adapt`, `genre_adapt`, `singing_style_adapt`,
`language_accent_adapt`) — thin wrappers picking sensible target modules
and rank per use case. This remains the *offline, deliberate* fine-tuning
path — contrast with §22 (self-learning), which is the *online,
incremental* path for smaller updates that don't warrant a full job.

---

## 21. Alignment — GRPO + DPO (`aarambh-voice-align`)

This is new relative to the draft, and it's the highest-value addition
because the reward infrastructure already exists in `-eval` (§24) — this
crate is mostly plumbing, not new research.

### 21.1 GRPO (Group Relative Policy Optimization)

```
For each prompt in a training batch:
  1. Sample K candidate generations (K=4-8) from the current policy
  2. Score each candidate with a weighted combination of -eval metrics:
       reward = w1·(1 - WER_normalized) + w2·speaker_sim + w3·emotion_acc
              + w4·music_tag_agreement + w5·MOS_proxy
     (weights depend on which engine/subsystem is being aligned — a TTS
      GRPO run weights WER and MOS_proxy heavily; a singing GRPO run adds
      pitch-accuracy; a music-generation GRPO run weights tag agreement)
  3. Compute group-relative advantage: A_i = reward_i - mean(reward_group)
  4. Policy gradient step, advantage-weighted, no separate value network
     needed (this is GRPO's whole appeal vs. PPO — cheaper on Kaggle GPUs)
```

### 21.2 DPO (Direct Preference Optimization)

Cheaper complement, no live sampling loop required at training time:

```
1. Offline: for each prompt, generate N candidates, score with -eval,
   take (highest-scoring, lowest-scoring) as (chosen, rejected) pairs —
   fully automatic, no human labeling needed to start
2. Standard DPO loss:
   L_DPO = -log σ( β · [ log π(chosen)/π_ref(chosen)
                        - log π(rejected)/π_ref(rejected) ] )
3. π_ref is the frozen pre-alignment checkpoint (same role as SFT
   checkpoint in text-model DPO recipes)
```

### 21.3 Where this sits in the pipeline
New phase, after Fine-Tuning Refinement (§20) and before the Evaluation
Harness pass that produces the release scorecard — the point is to align
*after* the model already fine-tunes well per-speaker/per-style, since
GRPO/DPO refine overall quality/naturalness, not identity-specific
adaptation (that's §20's and §22's job).

### 21.4 Training schedule (reference, DPO, Small scale, Kaggle T4)
```toml
beta              = 0.1
learning_rate     = 5e-6
optimizer         = "AdamW"
max_steps         = 5000
batch_size        = 4
pairs_per_prompt  = 1        # top-vs-bottom of N=6 sampled candidates
device            = "cuda:0"
dtype             = "bf16"
```

---

## 22. Self-Learning (`aarambh-voice-selflearn`)

Full design lives in **`SELF_LEARNING_VOICE_STUDIO.md`** — this section is
a pointer plus the one-paragraph summary needed to understand where it
fits in the workspace.

In short: mirrors Manas's associative memory + gradient orthogonalization
anti-forgetting design, applied to speaker/style adaptation. Lets the
system absorb a new voice, singing style, or user correction as a small,
confidence-gated incremental update — checked against `-eval` before being
committed, rolled back automatically if it regresses quality — rather than
requiring a full `-finetune` job for every new voice. Called from
`-serve` at inference time for online adaptation; sits at Layer 5 next to
`-finetune` and `-align`.

---

## 23. Safety Layer (`aarambh-voice-safety`)

- **Consent gating** — every cloning request (Voice Engine §8, and its
  reuse in Singing Engine §13) requires `ConsentSpec` to be present and
  valid; requests without it fail closed, not open.
- **Inaudible watermarking** — every generated output, regardless of
  engine, carries an inaudible watermark identifying it as AI-generated.
  Same principle as industry watermarking approaches for generated audio;
  implemented from scratch here (spread-spectrum watermark in an
  inaudible frequency band, survives the mix/mastering stage in §14).
- **Guardrails** — text-input filtering before generation (not a
  generative-content moderation model, just pattern-level filtering
  consistent with the rest of this project's safety posture).

---

## 24. Evaluation Harness + Baseline Comparison (`aarambh-voice-eval`)

| Metric | Measures | Method |
|---|---|---|
| ASR round-trip WER | TTS/singing intelligibility | Transcribe generated audio, compare to source text |
| Speaker similarity | Cloning fidelity | Cosine similarity, reference vs. generated speaker embeddings |
| Emotion accuracy | Emotion control fidelity | Emotion classifier (§10) on generated output |
| Music tag agreement | Generation-to-prompt fidelity | §11 classifier on generated output vs. prompt |
| MOS proxy | Overall naturalness | Learned quality predictor trained on public MOS-labelled datasets |
| **Baseline delta** *(NEW)* | Are you actually closing the gap? | Same prompts run against fixed open-source reference checkpoints (Voice Engine: compare against open TTS baselines from the current leaderboard generation; Composer/Music Engine: compare against the current open-source full-song-generation reference) — re-run every release, tracked over time, not just measured once |

`Scorecard::to_markdown()` / `to_json()` produce the release report used
in Phase 23-equivalent (see Roadmap).

---

## 25. Crate-by-Crate Reference — 24 Crates

| Crate | Layer | Responsibility |
|---|---|---|
| `aarambh-voice-core` | 0 | Config types, request/response types, errors |
| `aarambh-voice-codec` | 1 | Transformer-bottleneck RVQ codec, 12.5Hz, semantic distillation |
| `aarambh-voice-data` | 1 | Dataset loaders, preprocessing, auto-labelling |
| `aarambh-voice-textprep` | 1 | G2P, text normalisation, Sanskrit/Hindi/English transliteration **[NEW]** |
| `aarambh-voice-nn` | 2 | Transformer block + conditioning injection |
| `aarambh-voice-kernel` | 2 | CPU SIMD kernels, CUDA prep, fused STFT |
| `aarambh-voice-model` | 3 | Per-engine models + diffusion refinement head |
| `aarambh-voice-weights` | 3 | SafeTensors save/load, checkpoint conversion |
| `aarambh-voice-train` | 4 | Pretraining and continued-training loops |
| `aarambh-voice-quant` | 4 | INT8 / INT4 / GGUF-style quantisation |
| `aarambh-voice-finetune` | 5 | LoRA / QLoRA / DoRA adapters, all engines |
| `aarambh-voice-align` | 5 | GRPO + DPO alignment **[NEW]** |
| `aarambh-voice-selflearn` | 5 | Online self-learning, anti-forgetting **[NEW]** |
| `aarambh-voice-speaker` | 5 | Zero-shot cloning, text-described voice design |
| `aarambh-voice-emotion` | 5 | Emotion embedding space, continuous intensity |
| `aarambh-voice-music` | 6 | Music understanding + background music generation |
| `aarambh-voice-sing` | 7 | Singing synthesis (AR + optional diffusion refinement) |
| `aarambh-voice-mix` | 7 | Vocal + instrumental mixing/mastering |
| `aarambh-voice-compose` | 8 | Structure planner + lyrics-to-song orchestrator |
| `aarambh-voice-safety` | 9 | Consent gating, watermarking, guardrails |
| `aarambh-voice-eval` | 9 | Evaluation harness + baseline comparison |
| `aarambh-voice-control` | 9 | Full control API / NaadRequest DSL |
| `aarambh-voice-inference` | 9 | Shared inference runtime, KV cache, speculative decoding |
| `aarambh-voice-serve` | 10 | HTTP inference server |
| `aarambh-voice-studio` (bin) | 11 | CLI |

---

## 26. Data Flow Across the Workspace

```
raw audio / text / lyrics
        │
        ▼
aarambh-voice-textprep ──► G2P, normalisation, transliteration
        │
        ▼
aarambh-voice-data ──► preprocessing, auto-labelling (via -music understanding)
        │
        ▼
aarambh-voice-codec ──► discrete audio tokens (frozen after Stage 0)
        │
        ▼
aarambh-voice-nn + aarambh-voice-model ──► conditioned transformer
        │            ▲            ▲            ▲
        │      speaker_emb   emotion_emb   melody/duration
        ▼
aarambh-voice-align ──► GRPO/DPO-aligned checkpoint
        │
        ▼
aarambh-voice-selflearn ──► online incremental updates (confidence-gated)
        │
        ▼
aarambh-voice-inference ──► streamed audio tokens (speculative decoding) ──► codec decode ──► waveform
        │
        ▼
aarambh-voice-compose (structure planner, if full-song) ──►
aarambh-voice-mix (if singing + music) ──►
aarambh-voice-safety (watermark) ──► output (§29 formats)
```

---

## 27. Memory & Compute Estimates

### Training Memory (BF16, per scale, transformer core only)

| Scale | Weights | Gradients | AdamW States | Activations | Total |
|---|---|---|---|---|---|
| Tiny   | 40 MB   | 40 MB   | 160 MB  | ~80 MB  | ~0.32 GB |
| Small  | 220 MB  | 220 MB  | 880 MB  | ~440 MB | ~1.76 GB |
| Medium | 680 MB  | 680 MB  | 2.7 GB  | ~1.4 GB | ~5.5 GB  |
| Large  | 1.8 GB  | 1.8 GB  | 7.2 GB  | ~2.9 GB | ~13.7 GB |

> Codec adds a roughly fixed ~150–300 MB regardless of transformer scale
> (unchanged from draft — the transformer bottleneck inside the codec is
> small by design). Trained separately (Stage 0), frozen thereafter.

### CPU Inference Memory (F32 weights + KV cache, 10s output)

| Scale | Weights | KV Cache | Codec | Total |
|---|---|---|---|---|
| Tiny   | 80 MB   | 12 MB  | ~150 MB | ~242 MB |
| Small  | 440 MB  | 40 MB  | ~150 MB | ~630 MB |
| Medium | 1.36 GB | 160 MB | ~200 MB | ~1.72 GB |
| Large  | 3.6 GB  | 320 MB | ~250 MB | ~4.17 GB |

### CPU Inference Memory (INT4 quantised + KV cache, 10s output)

| Scale | Weights (Q4) | KV Cache | Codec | Total |
|---|---|---|---|---|
| Tiny   | 11 MB  | 12 MB  | ~150 MB | ~173 MB |
| Small  | 58 MB  | 40 MB  | ~150 MB | ~248 MB |
| Medium | 178 MB | 160 MB | ~200 MB | ~538 MB |
| Large  | 470 MB | 320 MB | ~250 MB | ~1.04 GB |

### Speculative decoding overhead **(NEW)**

Holding a Tiny draft checkpoint alongside the target adds a small,
roughly fixed memory cost regardless of target scale:

| Target scale | + Draft (Tiny, INT4) | New total (INT4 inference) |
|---|---|---|
| Small  | +11 MB | ~259 MB |
| Medium | +11 MB | ~549 MB |
| Large  | +11 MB | ~1.05 GB |

Negligible memory cost for a 1.5–2.5x latency win — this is why pulling
speculative decoding into v1 (§17.3) is worth the added complexity.

---

## 28. Hardware Strategy

### Your Local Machine (i3-1115G4, 8 GB RAM, Pop OS)

**Use exclusively for Tiny scale:**
- Codec smoke-training on LJSpeech excerpt
- Full Tiny TTS training loop, structure-planner training (small model,
  cheap even at Tiny)
- All unit and integration tests
- CLI inference (`speak`, `clone`, `design`, `compose`) with Tiny checkpoints
- QLoRA fine-tuning of Small on a single speaker's data
- INT4 inference of Medium
- Self-learning online updates at inference time (cheap by design — see
  `SELF_LEARNING_VOICE_STUDIO.md`)

### Kaggle GPU

| Scale | GPU | Dtype | Batch | Notes |
|---|---|---|---|---|
| Small  | T4 16 GB   | BF16 | 8–16 | Codec + Voice Engine, DPO alignment |
| Medium | P100 16 GB | BF16 | 4  | Music Engine generation (heaviest phase), GRPO alignment |
| Large  | A100 40 GB | BF16 | 2  | Stretch goal, hardware-dependent |

Same opt-in `--features cuda` pattern as `aarambh-ai`: CPU builds remain
default.

### Which phases actually need Kaggle

| Subsystem | Realistic on i3 alone? |
|---|---|
| Voice Engine (TTS, cloning, design, emotion) | Yes, Tiny/Small; Kaggle for Medium+ |
| Music Engine — understanding | Yes on i3 |
| Music Engine — generation | No — Kaggle required, heaviest phase |
| Singing Engine (Stage A) | Kaggle recommended, i3 for Tiny smoke tests |
| Singing Engine (Stage B, diffusion refinement) | No — Kaggle required |
| Composer + structure planner + Control + Mix | Yes on i3 — integration/small-model work |
| Alignment (GRPO) | No — needs sampling K candidates per step, Kaggle |
| Alignment (DPO) | Small scale possible on i3 with patience; Kaggle recommended |
| Self-learning (online updates) | Yes on i3 by design — see dedicated doc |

---

## 29. Audio Output Formats

New section — the draft had no explicit output-format decision.

| Format | Encoder crate | Use case | Default? |
|---|---|---|---|
| WAV (PCM16/24/32f) | `hound` (already a dependency) | Lossless, universal compatibility, CLI default output | **Yes, default** |
| FLAC | `flacenc` | Lossless, smaller than WAV, good for archival/download | Available, off by default |
| Opus | `audiopus` | Streaming server (§17.2 chunked streaming), lowest bandwidth for acceptable quality at speech/music bitrates | Default for `-serve` streaming responses |
| MP3 | `mp3lame-encoder` (feature `mp3`) | Broad legacy compatibility only | Off by default — LAME's licensing terms mean this stays feature-gated, documented clearly in the crate README so downstream users make an informed choice |

CLI default: `aarambh-voice-studio speak --out output.wav` (WAV, PCM16,
matching the sample rate of whichever scale/domain was used — 24kHz
speech, 44.1kHz music/singing where the mixing stage upsamples).
Server default: chunked Opus over HTTP for `/generate`, with a
`?format=wav` query override for clients that want lossless.

---

## 30. Relationship to `aarambh-ai`

Unchanged from the draft — sibling project, not a fork:
- Transformer block ported, not shared as a direct dependency (§7.1).
- Training, fine-tuning, quantisation, alignment, and safety-layer
  discipline are directly modelled on `aarambh-ai`'s equivalent crates —
  including the alignment crate, which follows the same GRPO recipe
  `aarambh-ai` v2.0 is separately planning, and the self-learning crate,
  which follows Manas's anti-forgetting design directly.
- Joint text+audio multimodal model remains out of scope for v1 (§31) —
  the two projects share *patterns*, not a runtime.

---

## 31. What's Explicitly Out of Scope (v1)

- Real-time (sub-100ms) conversational-latency streaming — v1 targets
  chunked streaming (§17.2) and speculative-decoding speedups (§17.3), not
  SSM-style linear-time architectures. Revisit only if conversational
  latency becomes an explicit goal — see the SSM note in the design
  rationale.
- Joint text+audio multimodal model combining `aarambh-ai` and
  `aarambh-voice-studio` into one transformer.
- Video or lip-sync generation.
- From-scratch pretraining of the Music Engine as the default path —
  adaptation is the plan; from-scratch is a stretch goal only.
- Multi-GPU distributed training (single-GPU Kaggle only).
- Pretrained checkpoints, adapters, or voice packs bundled in the
  repository — source/engineering release only.
- MP3 encoding on by default (licensing — kept behind a cargo feature,
  see §29).
