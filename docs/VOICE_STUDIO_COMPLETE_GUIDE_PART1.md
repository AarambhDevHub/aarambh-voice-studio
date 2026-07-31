# aarambh-voice-studio: The Complete Beginner's Guide (Part 1 of 2)

### Everything we're building, in plain human language

This document explains, step by step, everything inside `aarambh-voice-studio`
— a from-scratch speech, music, and singing AI studio built in Rust using
Candle. It covers Phases 0 through 27 across the full roadmap. Think of
this as a story: each phase builds on top of the one before it, like
constructing a building floor by floor.

No prior audio AI knowledge assumed. Every section has:
- A **plain-English definition**
- A **beginner explanation**
- **Why we actually need it**
- A **real-world example**
- A **diagram**
- **Common beginner questions**

This file covers Phases 0–13. Part 2 covers Phases 14–27.

---

## The Big Picture First

Before diving into 28 phases, here's the one-sentence version of what this
whole project actually is:

> `aarambh-voice-studio` is a set of models that turn sound into short
> lists of numbers (tokens), and then predict "what token comes next" —
> for speech, for music, and for singing — the exact same trick
> `aarambh-studio` uses for text, applied to audio instead of words.

Here is the full pipeline, zoomed way out, so you can see where every
phase fits:

```
 RAW AUDIO + TEXT
      │
      ▼
┌─────────────┐
│    CODEC    │  (turns waveforms into tokens, and back again)
└─────────────┘
      │
      ▼
┌─────────────┐
│  TEXT PREP  │  (turns written words into sounds-to-say)
│  + DATA     │
└─────────────┘
      │
      ▼
┌─────────────┐
│   NEURAL    │  (the shared "brain" - layers of math, same as aarambh-studio)
│   NETWORK   │
└─────────────┘
      │
      ▼
┌─────────────┐
│  TRAINING   │  (the model practices on real speech, music, and singing)
│    LOOP     │
└─────────────┘
      │
      ▼
┌─────────────┐
│  INFERENCE  │  (the trained model generates new audio for you)
│   ENGINE    │
└─────────────┘
```

Everything else — cloning, emotion, music, singing, mixing, composing,
alignment, self-learning — are **engines and upgrades** built on top of
this core pipeline. Keep this diagram in your head as we go.

---

## Phase 0: Workspace + Core Types

**Definition:** The empty skeleton of the project — folders, shared config
types, shared error types — with nothing "smart" in it yet.

**Beginner explanation:**
Before you can train anything, you need a place to put it and a shared
language for describing it. `ModelConfig` is that shared language: it
says "a Tiny model uses 256 numbers per token, processes things through 6
layers, and pays attention using 8 heads at once." Every later phase
reads these numbers instead of each inventing its own.

**Why we need it:**
Without one shared config type, every crate would invent its own way to
describe "how big is the model," and they'd drift out of sync — Phase 6's
training loop and Phase 25's GPU scale-up need to be talking about
*exactly* the same thing when they both say "Small scale."

**Example:**
```
let cfg = ModelConfig::tiny(AudioDomain::Speech);

cfg.d_model  = 256   (how many numbers describe one token)
cfg.n_layers = 6     (how many processing layers stacked)
cfg.n_heads  = 8     (how many attention "spotlights" run at once)
```

**Diagram:**
```
   ModelConfig::tiny()
          │
          ▼
  ┌─────────────────┐
  │ d_model  = 256   │
  │ n_layers = 6     │
  │ n_heads  = 8     │
  └─────────────────┘
          │
          ▼
  Used identically by Phase 6 (training),
  Phase 7 (inference), Phase 25 (GPU scale-up)
```

**Common beginner questions:**
- *Q: Why does "Tiny" even exist if we want a good final model?* → Tiny
  exists so the whole pipeline can be proven correct on an ordinary
  laptop, in minutes, before spending any Kaggle GPU time on a bigger
  scale that would take hours to fail if something were wrong.
- *Q: Could different crates just agree informally without a shared
  type?* → In theory, but in practice that's exactly how "it works on my
  machine" bugs happen — one crate's idea of "Small" quietly drifting
  from another's.

---

## Phase 1: Neural Audio Codec

**Definition:** A system that compresses a sound wave down into a short
list of numbers (tokens), and can reconstruct a very close copy of the
original sound from those same numbers.

**Beginner explanation:**
A raw waveform is enormous — one second of audio is 24,000 individual
numbers at 24kHz. A transformer can't reasonably chew through that
directly, the same way you wouldn't read a book letter-by-letter if you
could read it word-by-word instead. The codec's job is to be the "letters
into words" step for sound: it compresses one second of audio down to
roughly 12-13 tokens (this project's target frame rate), each token drawn
from a fixed dictionary of a few thousand possible sound-chunks.

There's a second, less obvious job this codec does: some of its tokens
are trained specifically to carry *meaning* (which words were said), not
just raw acoustic detail — a trick called semantic distillation. This
matters because it makes the *next* phase's job (predicting tokens with a
transformer) meaningfully easier.

**Why we need it:**
Every later engine — TTS, cloning, music, singing — works by predicting
"what token comes next," exactly like `aarambh-studio` predicts the next
word. None of that is possible without a codec that can turn tokens back
into audio a human can actually listen to.

**Example:**
```
waveform: [0.02, -0.01, 0.05, 0.03, -0.02, ...]   (24,000 numbers, 1 second)
                    │
                    ▼  Codec encoder + quantizer
tokens:   [451, 12, 998, 12, 87, ...]              (~13 numbers, 1 second)
                    │
                    ▼  Codec decoder
reconstructed waveform: [0.021, -0.009, 0.048, ...] (very close to original)
```

**Diagram:**
```
  original waveform
        │
        ▼
  ┌───────────┐
  │  Encoder  │  (shrinks it down)
  └───────────┘
        │
        ▼
  ┌───────────┐
  │  RVQ       │  (snaps to nearest "dictionary" entries)
  │  tokens    │
  └───────────┘
        │
        ▼
  ┌───────────┐
  │  Decoder  │  (expands it back)
  └───────────┘
        │
        ▼
  reconstructed waveform (should sound almost identical)
```

**Common beginner questions:**
- *Q: Is this the same thing as an MP3?* → Related idea (both compress
  audio), very different goal. MP3 is designed to be small and sound good
  to a human ear. This codec is designed to produce tokens that a
  *transformer* can easily learn to predict — good for modeling, not just
  good for listening.
- *Q: Why 12.5 tokens per second and not more?* → Fewer tokens per second
  means shorter sequences for the transformer to process, which means
  faster training and inference — especially important on an i3 laptop
  with no dedicated GPU.
- *Q: What happens if the codec is bad?* → Everything built on top
  inherits that badness — this is why the codec is frozen and never
  touched again once it passes its quality bar (the "freeze criterion"),
  the same discipline `aarambh-studio` uses for its tokenizer.

---

## Phase 2: Text Prep — G2P + Normalisation

**Definition:** Converting written text into the actual speech sounds it
represents, and converting things like "3" or "Dr." into their
spoken-out-loud form first.

**Beginner explanation:**
Two separate jobs live in this phase. **Normalisation** turns "Dr. Smith
met on 3/7" into "Doctor Smith met on the seventh of March" — the way
you'd actually say it out loud, not the way it's written. **G2P
(Grapheme-to-Phoneme)** then turns those words into phonemes — the actual
sound units a mouth makes — because English spelling doesn't reliably
tell you how to pronounce a word ("read" is pronounced two different ways
depending on tense; "colonel" doesn't sound anything like it's spelled).
For this project specifically, this phase also has to handle
Hindi/Sanskrit/English mixed within a single sentence, routing each
word's sounds through the correct language's rules.

**Why we need it:**
Without normalisation, the model would have to somehow learn, purely from
examples, that "3/7" sometimes means a date and sometimes a fraction —
an unnecessary and error-prone burden to place on a generative model.
Without G2P, the model has to guess pronunciation purely from spelling,
which fails constantly on names, borrowed words, and homographs (words
spelled the same but pronounced differently).

**Example:**
```
Input:      "मेरा नाम Darshan है, phase 2 पूरा हो गया"

Normalize:  numbers/dates converted to spoken form where present
            ("2" → "two" if it's meant to be spoken)

G2P:        [Hindi phonemes: मेरा नाम]
          + [English phonemes: Darshan]
          + [Hindi phonemes: है]
          + [English phonemes: phase two]
          + [Hindi phonemes: पूरा हो गया]
```

**Diagram:**
```
  written text (mixed languages, numbers, abbreviations)
          │
          ▼
  ┌───────────────┐
  │  Normalize     │  (numbers/dates/abbreviations → spoken form)
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  Code-switch   │  (detect which language each span is in)
  │  detection     │
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  G2P per span  │  (correct pronunciation rules per language)
  └───────────────┘
          │
          ▼
  phoneme sequence, ready for the TTS model
```

**Common beginner questions:**
- *Q: Why not just let the model figure out pronunciation on its own from
  raw text?* → Some systems do, but it makes the model's job much harder
  and less reliable, especially for languages mixed in one sentence — G2P
  does the "obvious but tedious" work up front so the model's capacity
  goes toward sounding natural, not toward guessing how to say a word.
- *Q: What happens with a completely made-up word the dictionary has
  never seen?* → A small learned fallback model (not just a fixed
  dictionary lookup) predicts likely phonemes from the spelling itself,
  the same way a human sounds out an unfamiliar word.

---

## Phase 3: Data Pipeline + Auto-Labelling

**Definition:** The machinery that loads raw audio files and their
metadata (text, speaker identity, genre, melody) into a clean, consistent
shape the training loop can actually use.

**Beginner explanation:**
Real datasets are messy — different sample rates, mono vs. stereo,
silence at the start of a clip, mismatched text encodings, occasional
corrupted files. This phase is the "wash and chop the vegetables before
cooking" step. It also includes **forced alignment** — a separate tool
(Montreal Forced Aligner) that listens to a recording and its transcript
together and figures out exactly which milliseconds correspond to which
phoneme, which later phases need as ground truth for how long each sound
should last.

**Why we need it:**
Training directly on raw, inconsistent files means every batch has
different shapes, different loudness, different sample rates — the model
spends its limited capacity compensating for inconsistency instead of
learning the actual task of speaking or singing well.

**Example:**
```
Raw VCTK speaker folder: files at 48kHz, some with long silence at the
start, transcripts in a slightly different text encoding per speaker

Pipeline steps:
1. Resample every file to the codec's 24kHz
2. Trim leading/trailing silence
3. Normalize text encoding
4. Run forced alignment: figure out exactly which frames correspond
   to which phoneme in the transcript
5. Write everything into a clean JSONL manifest, ready for the loader
```

**Diagram:**
```
  raw audio + raw transcript
          │
          ▼
  ┌───────────────┐
  │  Validate      │  (exists? readable? what sample rate?)
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  Resample +    │
  │  trim silence  │
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  Forced        │  (which frames = which phoneme)
  │  alignment     │
  └───────────────┘
          │
          ▼
  clean, ready-to-train manifest entry
```

**Common beginner questions:**
- *Q: Why is forced alignment a separate tool instead of something the
  model learns itself?* → It's possible to learn duration jointly, but
  using a dedicated, proven alignment tool up front gives a much more
  reliable ground truth to train against, especially early in the
  project — the model's duration-prediction head (Phase 6) is trained
  *against* this ground truth, not trying to discover it from scratch.
- *Q: What happens to a corrupted audio file?* → It's logged and skipped,
  never allowed to crash the whole pipeline — a single bad file
  shouldn't be able to take down a training run that's been going for
  hours.

---

## Phase 4: NN Primitives + Conditioning Injection

**Definition:** The actual transformer building blocks (attention,
normalization, feed-forward layers), plus the wiring that lets extra
information — "make it sound like this speaker," "make it sound happy" —
steer what gets generated.

**Beginner explanation:**
The transformer core itself (RMSNorm, RoPE, Grouped-Query Attention,
SwiGLU) is the exact same kind of machinery `aarambh-studio` uses for text —
full explanations of each live in the math guide. What's new here is
**conditioning injection**: a speaker's identity gets turned into a list
of 256 numbers (an embedding) and added into the very first layer, so
every later layer of processing already "knows" whose voice to generate.
Emotion works a bit differently — injected partway through the stack via
cross-attention, because emotion is more like a running commentary
throughout a sentence than a single fixed fact about it.

**Why we need it:**
Without a defined injection mechanism, "make this sound like speaker X"
would have no path into the model at all — the transformer would have no
way of knowing it's even supposed to differ from any other generation.

**Example:**
```
Same text: "Hello there"
Same transformer weights

speaker_embedding_A = [0.1, 0.9, -0.3, ...]  → generates in voice A
speaker_embedding_B = [0.8, -0.2, 0.4, ...]  → generates in voice B

(injected at layer 0, broadcast-added to every token's representation)
```

**Diagram:**
```
  token embeddings
        │
        ▼ + speaker_embedding (added at layer 0)
  ┌───────────────┐
  │ Transformer    │ ◄── emotion_embedding (cross-attended partway through)
  │ blocks × N     │
  └───────────────┘
        │
        ▼
  next-token prediction (steered by both speaker and emotion)
```

**Common beginner questions:**
- *Q: Why inject speaker identity at the very first layer but emotion
  partway through?* → Speaker identity is a single fixed fact about the
  whole utterance (broadcasting it once, early, is enough); emotion needs
  to keep influencing generation as the sentence unfolds, so it's
  injected via cross-attention at multiple points instead of just once.
- *Q: Could you add a third kind of conditioning later (like accent)?* →
  Yes — this is exactly why the injection mechanism was designed as a
  general pattern instead of something hard-coded only for speaker and
  emotion.

---

## Phase 5: CPU SIMD Kernels + CUDA Prep

**Definition:** Hand-optimized versions of the model's most frequently-run
math operations, so training and inference run faster on ordinary
hardware.

**Beginner explanation:**
A correct, naive implementation of attention or spectrogram computation
(STFT) works, but leaves a lot of speed on the table. Modern CPUs can do
several numbers' worth of math in a single instruction (SIMD — Single
Instruction, Multiple Data) if the code is written to take advantage of
it. This phase takes the "correct but slow" code from earlier phases and
gives it a faster, hand-tuned twin — with the original always kept as a
safe fallback.

**Why we need it:**
On an i3 laptop with no dedicated GPU, the difference between naive and
SIMD-optimized code is often the difference between a training run
finishing in an evening versus a training run that realistically never
finishes.

**Example:**
```
Naive attention: process one number at a time
  for i in 0..768 { result[i] = a[i] * b[i] }        → 768 individual steps

SIMD attention: process 8 numbers at once (example width)
  for i in (0..768).step_by(8) { result[i..i+8] = simd_multiply(...) }
                                                        → ~96 steps instead
```

**Diagram:**
```
  naive kernel  ──►  correct, but slow, 768 individual operations
       │
       ▼ (fallback path, always tested and kept)
  SIMD kernel   ──►  same math, ~8x fewer steps, same correct result
       │
       ▼
  benchmarked with criterion to confirm the speedup is real
```

**Common beginner questions:**
- *Q: Is the SIMD version "less accurate" since it's faster?* → No — it
  computes the exact same math, just several numbers at a time instead of
  one at a time. Tests confirm its output matches the naive version
  within a tiny floating-point tolerance.
- *Q: Why keep the slow version at all once the fast one exists?* → As a
  safety net — if the SIMD path isn't available on some CPU, or a bug is
  suspected, the naive path is a known-correct reference to fall back to
  or compare against.

---

## Phase 6: TTS Baseline — Tiny Trains!

**Definition:** The first moment the whole pipeline runs end-to-end: text
goes in, recognizable speech comes out, from a model trained by you.

**Beginner explanation:**
This phase wires Phases 1-5 together into one real training loop. Text
is phonemized (Phase 2), phonemes are fed through the transformer (Phase
4), which predicts codec tokens (Phase 1) one at a time, and the training
loss compares predicted tokens against the real ones taken from an actual
human recording. This is this project's version of `aarambh-studio`'s first
"it generates real words" milestone.

**Why we need it:**
This is the proof that the whole foundation actually works, before any of
the more specialized engines (cloning, music, singing) get built on top —
every one of those engines reuses this exact training recipe with extra
conditioning added on top.

**Example:**
```
Dataset: LJSpeech (one speaker, ~24 hours of clean audiobook recordings)

Step 1000:  loss = 4.2   (mostly guessing, output is noise-like)
Step 5000:  loss = 1.8   (rough speech-like sounds emerging)
Step 10000: loss = 0.9   (recognizably says the right words)

aarambh-voice-studio speak --text "hello world" --scale tiny --out out.wav
  → produces audio a human recognizes as "hello world"
```

**Diagram:**
```
  text  ──► Phase 2 (phonemes) ──► Phase 4 (transformer)
                                          │
                                          ▼ predicts codec tokens
                              compare against real recording's tokens
                                          │
                                          ▼
                                    training loss
                                          │
                                          ▼
                              adjust model weights, repeat
```

**Common beginner questions:**
- *Q: Why LJSpeech specifically for this first training run?* → It's a
  single speaker with many clean hours of recordings — depth over
  breadth is exactly what a first "does the pipeline actually work" test
  needs, since you're not yet trying to generalize across many voices.
- *Q: How do you know 0.9 loss is "good enough"?* → Loss alone isn't the
  full picture — this phase's actual milestone is measured by
  intelligibility: transcribing the generated audio back to text (ASR)
  and checking it matches what was asked for, within a fixed error-rate
  threshold.

---

## Phase 7: Inference Engine + CLI

**Definition:** The machinery that runs a trained model efficiently at
generation time, plus the actual command-line tool a person types into.

**Beginner explanation:**
Training and inference (generating new audio from a finished model) have
different needs. Inference needs to be fast, and needs a **KV cache** — a
memory of what's already been computed for tokens generated so far, so
the model doesn't redo the same work for every single new token. This
phase also scaffolds **speculative decoding** (a small "draft" model
guesses several tokens ahead, a bigger model checks them all at once) —
not fully active yet (that's Phase 25), but the plumbing is built now so
it's ready to switch on later.

**Why we need it:**
Without a KV cache, generating a 10-second clip would mean redoing the
entire computation from scratch for every single new token — wildly
slower than necessary, and the difference between a CLI that feels
responsive and one that doesn't.

**Example:**
```
Without KV cache: generating token #500 recomputes attention over
  all 499 previous tokens from scratch, every single time

With KV cache: attention results for tokens 1-499 are already stored;
  generating token #500 only computes the new part
```

**Diagram:**
```
  aarambh-voice-studio speak --text "..." --out demo.wav
          │
          ▼
  ┌───────────────┐
  │  CLI (clap)    │  parses the command
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  Inference     │  ◄── KV cache (remembers past computation)
  │  engine        │
  └───────────────┘
          │
          ▼
  demo.wav written to disk
```

**Common beginner questions:**
- *Q: What's the actual downside of not having a KV cache?* → Generation
  time would grow roughly with the square of the output length instead
  of linearly — a 10-second clip could take dramatically longer than a
  5-second one, instead of roughly twice as long.
- *Q: Why scaffold speculative decoding now if it's not used until Phase
  25?* → Building the interface early means later phases (cloning, music,
  singing) are written against a stable inference API from day one,
  rather than needing to be rewritten later when speculative decoding
  actually gets switched on.

---

## Phase 8: Voice Cloning (Zero-Shot)

**Definition:** Making the model speak in a specific person's voice using
only a short (3-10 second) sample of that voice, with no retraining
required.

**Beginner explanation:**
A separate small model (the speaker encoder) listens to the reference
clip and produces a 256-number "voice fingerprint" — the same kind of
embedding used for conditioning in Phase 4. "Zero-shot" means the main TTS
model has never specifically heard this person before; it's generalizing
from having seen thousands of *other* voices during training, the same
way a person can often mimic a new accent reasonably well after hearing
just a few sentences of it, without ever having met that specific speaker
before.

**Why we need it:**
Without zero-shot cloning, every new voice would require a full
retraining pass — completely impractical for any real product, and the
reason a dedicated speaker encoder (trained with GE2E loss, see math
guide) exists as its own small model.

**Example:**
```
Reference clip: 8 seconds of someone speaking
        │
        ▼ speaker encoder
speaker_embedding = [0.12, -0.45, 0.88, ..., 0.03]   (256 numbers)
        │
        ▼ fed into TTS model alongside new text
"aarambh-voice-studio clone --text 'new sentence' --reference ref.wav --out cloned.wav"
        │
        ▼
new speech, in that person's voice, saying something they never actually said
```

**Diagram:**
```
  reference audio (3-10 sec)
          │
          ▼
  ┌───────────────┐
  │ Speaker        │  produces a 256-number fingerprint
  │ encoder        │
  └───────────────┘
          │
          ▼
  speaker_embedding ──► injected into TTS model (Phase 4's mechanism)
          │
          ▼
  new text  ──►  generated speech, in the reference speaker's voice
```

**Common beginner questions:**
- *Q: Does the model "remember" this speaker permanently after one
  clone?* → No — a zero-shot clone only uses the embedding for that one
  request. Making the system *remember* a voice for future use without
  re-uploading a reference clip every time is what Phase 23's
  self-learning system is for.
- *Q: Can this be misused?* → Yes, which is exactly why Phase 19's
  consent gating and watermarking exist — cloning is never allowed to
  happen without an explicit consent token attached to the request.

---

## Phase 9: Voice Design (Text-Described Voice)

**Definition:** Creating a synthetic voice from a written description
("deep, warm, older male voice") instead of a reference recording.

**Beginner explanation:**
This reuses the exact same 256-number embedding space from Phase 8's
cloning — but instead of listening to audio to produce the embedding, a
small model reads the text description and *predicts* where in that same
256-number space such a voice would land. It's trained so that voices
whose real speakers were described similarly by humans end up embedded
near each other — meaning a brand-new text description can be projected
into a sensible spot even for a voice that's never actually existed.

**Why we need it:**
Not every use case has a reference recording available — sometimes you
want "a voice" that's never belonged to a real person at all, and this
phase is what makes that possible without collecting a new speaker's real
audio.

**Example:**
```
Text description: "bright, expressive female voice, slightly young-sounding"
        │
        ▼ projection MLP
synthetic_speaker_embedding = [0.34, 0.71, -0.12, ..., 0.55]

(lands near the real embeddings of speakers whose voices were
 human-described with similar words during training)
        │
        ▼
fed into TTS model exactly like a cloned embedding would be
```

**Diagram:**
```
  "deep, warm, older male voice"
          │
          ▼
  ┌───────────────┐
  │  Text encoder  │  (shared with textprep's embedding stack)
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  Projection    │  maps text meaning → point in speaker-embedding space
  │  MLP           │
  └───────────────┘
          │
          ▼
  synthetic speaker_embedding ──► used exactly like Phase 8's cloned embedding
```

**Common beginner questions:**
- *Q: Is a "designed" voice the same as an existing real speaker?* → Not
  necessarily — it lands in a *plausible* region of the embedding space
  based on the description, but doesn't need to match any single real
  speaker exactly.
- *Q: Where does the (text description, voice) training data come from?*
  → Human-written (or LLM-assisted, human-reviewed) short descriptions of
  each speaker in the cloning corpus — see the terminology & dataset
  guide for the full pipeline.

---

## Phase 10: Emotion Control System

**Definition:** The ability to make speech sound happy, nervous, calm,
angry — with a dial for intensity, not just an on/off switch.

**Beginner explanation:**
Instead of picking from a fixed list like `[happy, sad, angry]`, this
phase builds a continuous 8-number "emotion space" (valence, arousal, and
six other learned dimensions) — meaning emotions can blend and vary in
strength, the way real emotional speech actually does ("a little
annoyed" and "furious" are different points along a line, not two
unrelated categories with nothing in between).

**Why we need it:**
Discrete emotion categories force awkward choices at the boundaries (is
"mildly amused" happy, or neutral?) and can't represent intensity at all.
A continuous space handles both naturally, and matches the
`--emotion calm:0.8` style intensity control the CLI exposes.

**Example:**
```
emotion_embedding for "nervous, intensity 0.3" = [0.1, 0.6, -0.2, ...]
emotion_embedding for "nervous, intensity 0.9" = [0.1, 0.9, -0.2, ...]
       (same "direction" in emotion space, different magnitude)
```

**Diagram:**
```
  text description OR discrete label + intensity
          │
          ▼
  ┌───────────────┐
  │ Emotion        │  produces an 8-number continuous embedding
  │ encoder        │
  └───────────────┘
          │
          ▼ cross-attended into the transformer (Phase 4)
  generated speech with matching prosody (rhythm, pitch, energy)
```

**Common beginner questions:**
- *Q: Why 8 numbers specifically?* → Two of them (valence — how
  positive/negative, and arousal — how calm/excited) are well-established
  in emotion research as capturing most of the variation; the other six
  are additional learned dimensions that pick up finer distinctions
  during training, rather than being hand-designed in advance.
- *Q: Can emotion change partway through a sentence?* → The architecture
  supports it (cross-attention happens at multiple points through the
  sequence), though v1's control layer treats emotion as one setting per
  request rather than exposing mid-sentence emotion curves.

---

## Phase 11: Music Understanding

**Definition:** Teaching a model to listen to music and describe it —
genre, tempo (BPM), musical key, mood, which instruments are present.

**Beginner explanation:**
This is a *classifier*, not a generator — closer in spirit to a spam
filter than to a text generator. It's built specifically *before* the
music generator in Phase 12, for a very practical reason: labelling a
large music dataset by hand (this is jazz, this is 120 BPM, this has a
saxophone) isn't realistic for one person to do at scale, so this
classifier does that labelling automatically, once it's been trained on a
smaller, human-labelled starter set.

**Why we need it:**
Phase 12's generator needs a large labelled dataset to learn "make
something that sounds like X" — this phase is what makes building that
dataset possible without months of manual tagging.

**Example:**
```
Input: 20-second music clip
        │
        ▼ Music understanding classifier
Output: genre = "lo-fi hip hop" (91% confidence)
        tempo = 82 BPM
        key   = "C minor"
        mood  = ["relaxed", "rainy", "nostalgic"]
        instrumentation = ["piano", "vinyl_crackle", "soft_drums"]
```

**Diagram:**
```
  music audio ──► mel spectrogram ──► classifier encoder
                                            │
             ┌──────────────┬──────────────┼──────────────┬──────────────┐
             ▼              ▼              ▼              ▼              ▼
          genre          tempo           key            mood       instruments
        (softmax)     (regression)    (softmax)     (multi-label)  (multi-label)
```

**Common beginner questions:**
- *Q: Why build the "listening" model before the "generating" model?* →
  Because the listening model becomes the tool that lets you build a
  huge, automatically-labelled dataset for the generator — trying to
  build the generator first would mean either hand-labelling everything
  or generating with no sense of what "on-style" even means.
- *Q: How accurate does this classifier need to be?* → Accurate enough
  that a human spot-check of its auto-labels looks broadly correct — it
  doesn't need to be flawless, but systematic errors would get amplified
  at the scale Phase 12 uses it.

---

## Phase 12: Background Music Generation ⚠ Heaviest Phase

**Definition:** Generating instrumental music from a text description,
using the same "predict the next token" approach as speech, but for
music.

**Beginner explanation:**
Same underlying mechanism as Phase 6's TTS — predict the next audio
token, over and over — but conditioned on a style prompt ("upbeat lo-fi
hip hop, 90 BPM") instead of text-to-be-spoken, and using the music
domain of the shared codec (Phase 1). This is flagged as the heaviest
phase in the whole roadmap because music has much more internal structure
to get right than a single spoken sentence: rhythm has to stay locked in
for the whole clip, harmony has to make sense as it develops over time,
and mistakes are often more obviously audible to a listener than a
slightly-off vowel sound would be in speech.

**Why we need it:**
This is the "Music Engine" half of the Song Composer (Phase 17) —
without it, `aarambh-voice-studio compose` would produce vocals with no
backing track at all.

**Example:**
```
aarambh-voice-studio music generate \
  --prompt "upbeat lo-fi hip hop, 90 BPM" --duration 20 --out beat.wav

  → 20 seconds of instrumental music, generated token by token,
    conditioned on the text prompt above, then decoded back to
    audio through Phase 1's codec
```

**Diagram:**
```
  "upbeat lo-fi hip hop, 90 BPM"
          │
          ▼
  ┌───────────────┐
  │ Text prompt    │
  │ embedding      │
  └───────────────┘
          │
          ▼ conditions the transformer
  ┌───────────────┐
  │ Autoregressive │  predicts music tokens, one at a time
  │ generation     │
  └───────────────┘
          │
          ▼ every 500 steps during training
  ┌───────────────┐
  │ Phase 11       │  checks: does this sound like the prompt asked for?
  │ classifier     │  (this becomes part of the training reward)
  └───────────────┘
```

**Common beginner questions:**
- *Q: Why is this specifically the "heaviest" phase?* → Speech has a
  natural, forgiving rhythm (however you say a sentence is roughly fine);
  music has to maintain a locked-in tempo and coherent harmony over the
  *entire* clip, which is a much harder pattern for a model to learn to
  sustain — and any drift is very audible.
- *Q: Does this need real music theory knowledge built in?* → No explicit
  music theory rules are hand-coded — the model learns tempo, key, and
  harmony patterns statistically from the auto-labelled training data,
  the same way `aarambh-studio` learns grammar without being taught explicit
  grammar rules.

---

## Phase 13: Singing Synthesis Stage A (A Cappella, Autoregressive)

**Definition:** Turning lyrics plus a melody (which notes, how long each
one lasts) into sung vocals.

**Beginner explanation:**
Singing is TTS's harder cousin. In ordinary speech, the model gets to
choose its own natural rhythm and pitch; in singing, it has to hit
*exact* notes and *exact* durations dictated by the melody input, while
still producing intelligible lyrics on top of that constraint. This phase
adds two new pieces of conditioning beyond ordinary TTS: a pitch target
(predicted F0 — fundamental frequency, i.e. how high or low the voice is
at each moment) and a duration that's dictated by the melody rather than
predicted freely, the way Phase 6's TTS predicts its own natural pacing.

**Why we need it:**
Without explicit pitch and duration conditioning, the model would just
speak the lyrics in whatever natural rhythm it prefers — recognizable as
speech, but not as singing along to a specific tune.

**Example:**
```
Lyrics: "happy birthday to you"
Melody (simplified): notes [C4, C4, D4, C4], each held for a set duration

Stage A output: audio where "hap-py" lands on the first two C4 notes at
the correct pitch and timing, "birth-" lands on D4, and so on — sung,
not merely spoken, in time with the given melody
```

**Diagram:**
```
  lyrics + melody (MIDI-like: pitch + duration per note)
          │
          ▼
  ┌───────────────┐
  │ Transformer    │  ◄── melody cross-attention (every layer)
  │ core           │  ◄── pitch auxiliary head (predicts F0)
  └───────────────┘
          │
          ▼
  codec tokens, decoded into sung audio matching the melody
```

**Common beginner questions:**
- *Q: What happens if the melody and the natural rhythm of the lyrics
  don't quite fit?* → The melody's duration wins — the model is trained
  to stretch or compress syllables to match the given note durations,
  the same way a real singer adapts phrasing to fit a tune.
- *Q: Why is this "Stage A" — is there more?* → Yes — Phase 14 (Part 2)
  adds an optional second pass that further improves naturalness. Stage A
  alone is already a complete, working Singing Engine; Stage B is an
  additive quality improvement on top.

---

*Continue to Part 2 for Phases 14–27: singing refinement, mixing, the
song composer's structure planner, safety, quantisation, fine-tuning,
GRPO/DPO alignment, self-learning, evaluation, GPU scale-up, and the
inference server.*
