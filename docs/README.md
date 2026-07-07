# aarambh-voice-studio — Docs

This folder holds the learning material behind `aarambh-voice-studio` —
a from-scratch speech, music, and singing AI studio built in Rust using
Candle. If you've looked at this repo and wondered *"okay but how does
any of this actually work?"*, start here.

These docs aren't API references or code comments. They're written for
someone coming in with **zero background in audio AI/ML** — a beginner
who codes but has never touched a neural audio codec, a transformer, or
a GAN before. The goal is that by the end of these files, you understand
not just *what* `aarambh-voice-studio` does, but *why* every piece exists
and *how* the math underneath it actually works.

---

## What's in this folder

### 1. `VOICE_STUDIO_COMPLETE_GUIDE_PART1.md` / `PART2.md`
**The full project walkthrough — every phase, explained.**

This covers all 28 phases of `aarambh-voice-studio`, from Phase 0
(workspace scaffolding) through Phase 27 (production release):

- **Part 1 (Phases 0–13):** Workspace + core types, Neural Audio Codec,
  Text Prep (G2P + normalisation), Data Pipeline + Auto-Labelling, NN
  Primitives + Conditioning Injection, CPU/CUDA Kernels, TTS Baseline,
  Inference Engine + CLI, Voice Cloning, Voice Design, Emotion Control,
  Music Understanding, Background Music Generation, Singing Synthesis
  Stage A.
- **Part 2 (Phases 14–27):** Singing Stage B (Diffusion Refinement),
  Mixing, Cloning/Emotion for Singing, Structure Planner + Song
  Composer, Full Control Layer, Safety + Watermarking, Quantisation,
  Fine-Tuning (LoRA/QLoRA/DoRA), Alignment (GRPO + DPO), Self-Learning,
  Evaluation Harness + Baseline Comparison, GPU Scale-Up + Speculative
  Decoding, Inference Server + Output Formats, Production Release v1.0.

Each phase includes a plain-English definition, a beginner explanation,
why it's needed, and a worked example. Read this first — it's the map of
the whole project.

### 2. `VOICE_STUDIO_MATH_FORMULAS_GUIDE_PART1.md` / `PART2.md`
**The math underneath every phase, explained from zero.**

Once you know *what* each phase does, these files explain the actual
formulas doing the work:

- **Part 1:** Waveforms & sample rate, STFT/mel spectrograms, VQ
  commitment loss, reconstruction loss, adversarial (GAN) hinge loss +
  feature matching, semantic distillation, scaled dot-product attention,
  RoPE, GQA, cross-entropy loss, GE2E speaker-embedding loss.
- **Part 2:** Duration loss, pitch (F0) loss, LUFS loudness matching,
  diffusion/flow-matching objectives, GRPO's group-relative advantage,
  DPO's preference loss, gradient orthogonalization (self-learning
  anti-forgetting), spread-spectrum watermarking, and quantisation.

Every formula comes with a symbol-by-symbol translation (so `Σ`, `β`,
`σ` stop looking scary) and a worked numeric example solved by hand,
step by step. Read this after the phases guide, whenever you want to
understand the actual arithmetic behind a specific phase.

### 3. `VOICE_STUDIO_AUDIO_ML_TERMINOLOGY_AND_DATASET_GUIDE.md`
**The foundation underneath everything — terminology and where the training data comes from.**

Two parts:
- **Part 1** untangles the audio-AI terminology soup: DSP vs. Audio ML
  vs. Generative Audio AI, core audio vocabulary (waveform, spectrogram,
  codec, token), speech/voice vocabulary (TTS, ASR, WER, phoneme,
  zero-shot cloning), music/singing vocabulary (BPM, key, F0, MIDI,
  LUFS), generative model vocabulary (autoregressive, diffusion,
  flow-matching, RVQ, GAN), and alignment/self-learning vocabulary
  (GRPO, DPO, LoRA family, catastrophic forgetting).
- **Part 2** walks through the practical pipeline of building each
  phase's training dataset: where speech/music/singing corpora come
  from, consent requirements specific to voice data, auto-labelling,
  cleaning, manifest/JSONL formatting, train/val/test splitting (by
  speaker, not just by utterance), and licensing/ethics.

Read this first if you're completely new to audio AI in general, or read
it alongside the other guides whenever a term or a data-related phase
comes up.

---

## Suggested reading order

If you're starting from zero:

```
VOICE_STUDIO_AUDIO_ML_TERMINOLOGY_AND_DATASET_GUIDE.md   →  understand the terminology + where data comes from
            │
            ▼
VOICE_STUDIO_COMPLETE_GUIDE_PART1.md                    →  understand what voice-studio builds (Phases 0–13)
            │
            ▼
VOICE_STUDIO_COMPLETE_GUIDE_PART2.md                    →  understand what voice-studio builds (Phases 14–27)
            │
            ▼
VOICE_STUDIO_MATH_FORMULAS_GUIDE_PART1.md               →  understand the math behind Phases 1–13
            │
            ▼
VOICE_STUDIO_MATH_FORMULAS_GUIDE_PART2.md               →  understand the math behind Phases 13–23
```

If you already know the basics and just want the project-specific
details, jump straight to `VOICE_STUDIO_COMPLETE_GUIDE_PART1.md` and use
the terminology/math guides as reference whenever a term or formula is
unfamiliar.

---

## Who this is for

- Anyone reading the `aarambh-voice-studio` codebase for the first time
  and wondering what a given crate actually does.
- Viewers of the Aarambh Dev Hub YouTube channel who want the written
  version of what's explained on-screen.
- Contributors who want to understand a phase deeply enough to help
  extend it — see `CONTRIBUTING.md` for how to get started.
- Future-me, six months from now, who forgot why a loss function was
  weighted a certain way.

No prior audio ML background is assumed anywhere in these files. If
something is still unclear after reading, that's a gap in the doc, not a
gap in you — feel free to open an issue.

---

## Keeping these docs updated

As phases ship (especially the alignment, self-learning, and singing
Stage B phases — the newest and least-proven parts of the roadmap),
these files get updated to match. If a phase's implementation changes
significantly from what's described here, the corresponding section in
the Complete Guide and Math Formulas Guide should be revisited.

---

## Support aarambh-voice-studio

If these docs or the project itself helped you, consider supporting the
work:

- ☕ [Buy Me a Coffee](https://www.buymeacoffee.com/aarambhdevhub)
- 💖 [GitHub Sponsors](https://github.com/sponsors/aarambh-darshan)
- 🎓 [Topmate](https://topmate.io/darshan_vichhi) — 1-on-1 mentoring and paid sessions
- 🪙 [Razorpay](https://razorpay.me/@aarambhdevhub) — for India-based support

Every bit helps keep this project — and the free educational content
around it — going.

---

*Part of the Aarambh Dev Hub ecosystem. Built with Rust, one phase at a time.*
