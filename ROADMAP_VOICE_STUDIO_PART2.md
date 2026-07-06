# ROADMAP_VOICE_STUDIO.md — Part 2 of 2 — aarambh-voice-studio

> Continues directly from Part 1, Phase 13. Read Part 1 first.

---

## Phase 14 — Singing Synthesis Stage B: Diffusion Refinement *(NEW)*

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
An optional, additive diffusion/flow-matching refinement pass on top of
Phase 13's Stage A output, per ARCHITECTURE_VOICE_STUDIO_PART2.md §13.1.
Feature-gated `diffusion-refine` — Stage A remains fully functional
without this phase.

### Tasks

**`aarambh-voice-sing`** (extends Phase 13):
```
[ ] src/stage_b/model.rs — small diffusion/flow-matching decoder,
    conditioned on Stage A's codec-token output
[ ] src/stage_b/train.rs — L_sing_B (PART2 §13.2): denoising/flow-
    matching objective, trained after Stage A is frozen for the
    checkpoint being refined (not jointly)
[ ] src/stage_b/sample.rs — reverse-diffusion (or flow-matching ODE)
    sampling loop, feature-gated `diffusion-refine`
```

### Tests
```
[ ] Stage B output's MOS-proxy score exceeds Stage-A-only output on the
    same input, on a held-out set (the refinement must actually help,
    not just add latency)
[ ] Disabling the `diffusion-refine` feature falls back cleanly to
    Stage-A-only output with no errors
```

### Milestone
`aarambh-voice-studio sing --lyrics "..." --melody melody.mid --refine --out sung_refined.wav`
produces audibly more natural singing than the Stage-A-only path, verified
by the MOS-proxy delta. Tag: `v0.1.0-phase14`

---

## Phase 15 — Singing + Music Mixing

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
Combine independently-generated vocal and instrumental stems into a
mixed, mastered output, per ARCHITECTURE PART2 §14.

### Tasks

**`aarambh-voice-mix`:**
```
[ ] src/model.rs — lightweight U-Net over STFT magnitude, phase via
    Griffin-Lim or small neural vocoder head
[ ] src/train.rs — L_mix (PART2 §14.2): L1 spectrogram loss + LUFS-
    loudness matching penalty
[ ] src/master.rs — Mix::combine(vocal_stem, instrumental_stem, MixSpec) -> waveform
```

### Tests
```
[ ] Mixed output's measured LUFS is within tolerance of the target LUFS
    in MixSpec
[ ] Mixing does not introduce audible clipping (peak-level check) across
    a range of input gain combinations
```

### Milestone
`aarambh-voice-studio mix --vocal sung.wav --instrumental music.wav --target-lufs -14 --out mixed.wav`
produces a properly-mastered combined track. Tag: `v0.1.0-phase15`

---

## Phase 16 — Cloning + Emotion Extended to Singing

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
Reuse the Phase 8 (cloning) and Phase 10 (emotion) conditioning paths
inside the Singing Engine — verifying the "built once, reused" design
philosophy actually holds end-to-end.

### Tasks

**`aarambh-voice-sing`** (extends Phase 13/14):
```
[ ] src/conditioning.rs — wire speaker_embedding and emotion_embedding
    into the Singing Engine's conditioning injection points (same
    mechanism as Voice Engine, per ARCHITECTURE PART1 §7.2)
[ ] src/train_extended.rs — fine-tune Stage A (and Stage B if enabled)
    with cloning/emotion conditioning active, small additional training
    run rather than from-scratch
```

### Tests
```
[ ] A cloned voice's singing output scores above a fixed speaker-
    similarity threshold, same metric as Phase 8's speech cloning
[ ] Emotion-conditioned singing produces measurably different prosody
    (matching Phase 10's speech-side test, applied to singing output)
```

### Milestone
`aarambh-voice-studio sing --lyrics "..." --melody melody.mid --reference ref.wav --emotion "joyful" --out cloned_emotional_singing.wav`
Tag: `v0.1.0-phase16`

---

## Phase 17 — Structure Planner + Song Composer

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
The full lyrics-to-song orchestrator, with the structure planner
(verse/chorus/bridge prediction + repeat detection) running first, per
ARCHITECTURE PART2 §15.

### Tasks

**`aarambh-voice-compose`:**
```
[ ] src/structure.rs — sequence-labelling transformer: per-line section
    label + repeat-of-earlier-chorus detection
[ ] src/structure_train.rs — L_structure (PART2 §15.2): cross-entropy +
    BCE, trained on lyric sheets with existing section-header markup
    (free supervision, no manual annotation needed)
[ ] src/orchestrate.rs — Composer::compose(lyrics, style_prompt) ->
    dispatches sections to Singing Engine (reusing cached audio for
    repeated choruses), dispatches style_prompt to Music Engine informed
    by structure, calls Mix to combine
```

### Tests
```
[ ] Structure planner correctly labels section boundaries on a held-out
    set of lyric sheets with known ground-truth markup
[ ] Repeated choruses are detected and reuse the same generated audio
    (verified by checking the orchestrator does not re-invoke the Singing
    Engine for a detected repeat)
```

### Milestone
`aarambh-voice-studio compose --lyrics song.txt --style "acoustic folk" --out full_song.wav`
produces a complete, structurally coherent song. Tag: `v0.1.0-phase17`

---

## Phase 18 — Full Control Layer

**Duration:** 5–7 days | **Hardware:** i3

### Goal
`NaadRequest` fully fleshed out (stubbed in Phase 0), one typed request
struct exposing every parameter across all engines, per ARCHITECTURE
PART2 §16.

### Tasks

**`aarambh-voice-control`:**
```
[ ] src/request.rs — full NaadRequest, VoiceSpec, EmotionSpec,
    SingingSpec, MusicSpec, MixSpec, AudioOutputFormat, ConsentSpec
[ ] src/dispatch.rs — Control::handle(NaadRequest) -> routes to the
    correct sequence of engine calls based on which optional fields are
    populated
[ ] src/validate.rs — request validation (e.g. SingingSpec present without
    lyrics is a validation error, not a silent no-op)
```

### Tests
```
[ ] Every documented field on NaadRequest round-trips through
    serde_json without loss
[ ] Dispatch correctly routes a TTS-only request, a full-song request,
    and every combination in between, verified against a fixture matrix
[ ] Invalid request combinations are rejected with a clear error, not a
    panic
```

### Milestone
A single `NaadRequest` JSON payload correctly drives any combination of
engines end-to-end. Tag: `v0.1.0-phase18`

---

## Phase 19 — Safety & Watermarking

**Duration:** 7–10 days | **Hardware:** i3

### Goal
Consent gating and inaudible watermarking wired into every generation
path, per ARCHITECTURE PART2 §23 — no cloning path exists that bypasses
consent, and no output leaves the system unwatermarked.

### Tasks

**`aarambh-voice-safety`:**
```
[ ] src/consent.rs — ConsentSpec validation, fail-closed if missing on
    any request that includes reference_audio
[ ] src/watermark.rs — spread-spectrum inaudible watermark, embed +
    detect, verified to survive the Phase 15 mixing/mastering stage
[ ] src/guardrails.rs — text-input pattern-level filtering before
    generation
```

### Tests
```
[ ] A cloning request without a valid consent flag fails, does not
    silently proceed
[ ] Watermark is recoverable from generated output after passing through
    the Phase 15 mixing pipeline
[ ] Watermark is inaudible — measured via a perceptual-difference metric
    against unwatermarked output, not just "presumed inaudible"
```

### Milestone
`aarambh-voice-studio clone --reference ref.wav --text "..." --out x.wav`
without a consent flag fails cleanly; with it, succeeds and the output is
watermark-verifiable. Tag: `v0.1.0-phase19`

---

## Phase 20 — Quantisation Stack

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
INT8/INT4/GGUF-style quantisation across all engines, for low-latency CPU
deployment.

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
    ARCHITECTURE_VOICE_STUDIO_PART2.md §27
```

### Milestone
`aarambh-voice-studio quantise --engine tts --scale medium --format int4`
produces a checkpoint running within the documented i3 memory budget.
Tag: `v0.1.0-phase20`

---

## Phase 21 — Fine-Tuning Refinement (LoRA / QLoRA / DoRA)

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
Unified deliberate fine-tuning path across all engines — the offline
counterpart to Phase 23's self-learning.

### Tasks

**`aarambh-voice-finetune`:**
```
[ ] src/lora.rs — LoRA adapter injection into aarambh-voice-nn blocks
[ ] src/qlora.rs — QLoRA (quantised base + LoRA adapters)
[ ] src/dora.rs — weight-decomposed DoRA adapters
[ ] src/recipes.rs — speaker_adapt(), genre_adapt(), singing_style_adapt(),
    language_accent_adapt()
```

### Tests
```
[ ] LoRA fine-tune on a single speaker improves speaker-similarity metric
    over the base model
[ ] QLoRA fine-tune fits within Small-scale memory budget on Kaggle T4
[ ] DoRA singing-style adapter is distinguishable from base singing style
    via a style-classification spot-check
```

### Milestone
`aarambh-voice-studio finetune --recipe speaker_adapt --data speaker_clips/ --out adapter.safetensors`
Tag: `v0.1.0-phase21`

---

## Phase 22 — Alignment: GRPO + DPO *(NEW)*

**Duration:** 10–14 days | **Hardware:** Kaggle

### Goal
Reward-aligned quality improvement across all generative engines, using
`aarambh-voice-eval` metrics as reward signals, per ARCHITECTURE PART2
§21. This is the highest-leverage new phase — the reward infrastructure
already exists, this phase is mostly plumbing plus the actual training
runs.

### Tasks

**`aarambh-voice-align`:**
```
[ ] src/reward.rs — RewardAdapter wrapping aarambh-voice-eval metrics
    into a single weighted scalar reward, per-engine weight presets
    (TTS, singing, music — see PART2 §21.1)
[ ] src/grpo.rs — group-relative sampling (K=4-8 candidates per prompt),
    group-relative advantage computation, policy-gradient update
[ ] src/dpo.rs — offline preference-pair construction (top-vs-bottom of
    N sampled candidates per prompt), DPO loss against a frozen
    reference checkpoint
[ ] configs/dpo_small.toml — matches PART2 §21.4 reference schedule
```

### Tests
```
[ ] GRPO training run shows monotonic improvement in the weighted reward
    over a fixed number of steps on a smoke-test subset
[ ] DPO-aligned checkpoint's MOS-proxy score exceeds the pre-alignment
    (SFT-only) checkpoint on a held-out prompt set
[ ] Alignment does not regress speaker-similarity or WER relative to
    pre-alignment checkpoint (a GRPO/DPO run optimizing naturalness
    should not silently break intelligibility or cloning fidelity)
```

### Milestone
`aarambh-voice-studio align --engine tts --method dpo --scale small --out aligned.safetensors`
produces a checkpoint that beats the pre-alignment baseline on the
weighted reward, with no regression on the guardrail metrics above.
Tag: `v0.1.0-phase22`

---

## Phase 23 — Self-Learning *(NEW)*

**Duration:** 7–10 days | **Hardware:** i3

### Goal
Online, confidence-gated self-learning — full design in
`SELF_LEARNING_VOICE_STUDIO.md`. This phase implements that document.

### Tasks

**`aarambh-voice-selflearn`:**
```
[ ] src/memory.rs — AssociativeMemory (SELF_LEARNING doc §4)
[ ] src/orthogonalize.rs — gradient orthogonalization (§5)
[ ] src/adapter_bank.rs — self-growing AdapterBank (§6)
[ ] src/commit.rs — confidence-gated commit loop (§7)
[ ] src/update.rs — online_update() entry point (§10)
```

### Tests
```
[ ] All tests listed in SELF_LEARNING_VOICE_STUDIO.md §15, including the
    50-speaker anti-forgetting regression test
```

### Milestone
Exactly the milestone in SELF_LEARNING_VOICE_STUDIO.md §16:
`aarambh-voice-studio learn --sample new_voice.wav --identity-hint "warm, mid-30s"`
commits or rejects correctly, and the anti-forgetting regression test
passes. Tag: `v0.1.0-selflearn` (same tag as the dedicated doc — this
phase and that document describe the same deliverable).

---

## Phase 24 — Evaluation Harness + Baseline Comparison

**Duration:** 7–10 days | **Hardware:** i3 + Kaggle

### Goal
The full evaluation harness from ARCHITECTURE_VOICE_STUDIO_PART2.md §24,
now including the fixed external-baseline comparison, producing a single
scorecard across all engines.

### Tasks

**`aarambh-voice-eval`:**
```
[ ] src/asr_roundtrip.rs — WER via ASR round-trip
[ ] src/speaker_sim.rs — cosine similarity, reference vs. generated
    speaker embeddings
[ ] src/emotion_acc.rs — emotion-classification accuracy on generated
    output
[ ] src/music_tags.rs — tag agreement between prompt and analyze() output
[ ] src/mos_proxy.rs — learned naturalness predictor
[ ] src/baseline.rs — fixed comparison harness against external
    open-source reference checkpoints, one per engine, re-run every
    release
[ ] src/report.rs — Scorecard, to_markdown(), to_json()
```

### Tests
```
[ ] Each metric produces stable, reproducible scores on a fixed fixture
    set
[ ] Scorecard aggregates all metrics into one report without missing
    fields for any engine
[ ] Baseline comparison runs against the configured reference
    checkpoints and reports a clear delta, not just raw numbers
```

### Milestone
`aarambh-voice-studio eval --all --with-baseline` produces a full
scorecard across Voice, Music, and Singing engines, including baseline
deltas. Tag: `v0.1.0-phase24`

---

## Phase 25 — GPU Scale-Up + Speculative Decoding

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
Small/Medium/Large checkpoints for every engine, CUDA kernels activated,
and speculative decoding fully implemented and benchmarked (scaffolded
back in Phase 7).

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

**`aarambh-voice-inference`** (completes Phase 7's stub):
```
[ ] src/speculative.rs — real dual-model implementation: Tiny draft model
    proposes tokens, target model verifies in one forward pass, accepts
    longest matching prefix (ARCHITECTURE PART2 §17.3)
```

### Tests
```
[ ] CUDA kernel output matches CPU reference within float tolerance
[ ] Training throughput meets or exceeds targets in ARCHITECTURE
    PART2 §28's Kaggle GPU table
[ ] Speculative decoding produces output identical to non-speculative
    generation (correctness first) and measurably faster wall-clock time
    on CPU inference (1.5-2.5x target, per PART2 §17.3)
```

### Milestone
Small and Medium checkpoints exist for every engine, trained on Kaggle,
and speculative decoding shows a verified speedup with no output
divergence. Tag: `v0.1.0-phase25`

---

## Phase 26 — Inference Server + Audio Output Formats

**Duration:** 7–10 days | **Hardware:** i3

### Goal
An HTTP server exposing `NaadRequest`/`NaadResponse`, with chunked
streaming, and all four output formats correctly wired per ARCHITECTURE
PART2 §29.

### Tasks

**`aarambh-voice-serve`:**
```
[ ] src/server.rs — axum routes, one endpoint per capability plus a
    unified `/generate` endpoint accepting a full NaadRequest
[ ] src/batching.rs — continuous batching across concurrent requests
[ ] src/streaming.rs — chunked-transfer streaming, defaulting to Opus
    (via `audiopus`), reusing aarambh-voice-inference's frame-by-frame
    decode
[ ] src/formats.rs — WAV (hound, default for CLI/non-streaming), FLAC
    (flacenc), Opus (audiopus, default for streaming), MP3
    (mp3lame-encoder, behind `mp3` feature) — dispatch based on
    AudioOutputFormat / `?format=` query param
[ ] src/session.rs — per-request KV cache + safety-layer pass-through
[ ] src/selflearn_hook.rs — async call into aarambh-voice-selflearn's
    online_update() when a request sets `learn_from_this: true`
```

**`aarambh-voice-studio` (bin):**
```
[ ] src/cmd/serve.rs — `aarambh-voice-studio serve --port 8080`
```

### Tests
```
[ ] `/generate` with a TTS-only request returns a valid streamed
    response in each of the four supported formats
[ ] Concurrent requests are batched without cross-request state leakage
[ ] Server rejects unwatermarkable / non-consented requests identically
    to the CLI path (safety layer is not bypassed by the server)
[ ] A request with `learn_from_this: true` triggers an async self-learning
    update without adding latency to the response
```

### Milestone
`aarambh-voice-studio serve --port 8080` accepts HTTP requests and
streams audio back for every engine, in every supported format.
Tag: `v0.1.0-phase26`

---

## Phase 27 — Production Release v1.0

**Duration:** 7–10 days | **Hardware:** all

### Goal
A tagged, documented, source-only v1.0.0 release — final version,
nothing added after this.

### Tasks
```
[ ] Strict docs pass across all 24 crates (crate-level doc comments,
    README per crate, donation info instead of contact details per
    project convention)
[ ] CI workflow: cargo check --workspace, cargo test --workspace,
    cargo clippy --workspace -- -D warnings
[ ] RELEASE.md — confirms: source release, crates use publish = false,
    no pretrained checkpoints/adapters/voice packs bundled
[ ] CHANGELOG_VOICE_STUDIO.md — full phase-by-phase history, all 28
    phases
[ ] Example Tiny configs and local smoke-test data paths documented for
    every engine
[ ] Final aarambh-voice-eval scorecard run (Phase 24), including baseline
    comparison, published alongside the release
[ ] Tag v1.0.0 release notes summarising all 24 crates and full
    capability list: TTS, cloning, voice design, emotion control, music
    generation/understanding, singing (AR + diffusion refinement),
    structure-aware song composition, GRPO/DPO alignment, self-learning,
    four audio output formats
```

### Tests
```
[ ] Full workspace builds and tests pass on a clean checkout with only
    documented prerequisites installed
[ ] Every CLI subcommand documented in the README runs successfully
    against a Tiny checkpoint on a clean i3-equivalent environment
[ ] Final scorecard shows no regression against any per-phase milestone
    metric captured earlier in the roadmap
```

### Milestone
```
git checkout v1.0.0
cargo build --release -p aarambh-voice-studio
target/release/aarambh-voice-studio --version
```
Tag: `v1.0.0`

Thanks for watching and happy coding!
