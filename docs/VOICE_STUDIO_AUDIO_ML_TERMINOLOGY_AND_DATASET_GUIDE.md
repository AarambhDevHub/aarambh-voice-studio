# Audio AI, ML & Dataset Creation: The Complete Beginner's Guide

### Understanding the terminology + how to actually build audio datasets, for aarambh-voice-studio

This guide is for someone who keeps hearing terms like "spectrogram,"
"codec," "zero-shot cloning," "GRPO" thrown around and wants to *actually*
understand them — not just buzzwords, but what they mean, why they exist,
and how they connect to building something like `aarambh-voice-studio`.

Same format as before, for every concept:
- **Definition**
- **Beginner explanation**
- **Why it matters**
- **Example**
- **Diagram**
- **Common beginner questions**

This file has two parts:
- **Part 1** untangles the terminology soup — from general signal
  processing all the way down to the specific techniques this project
  uses.
- **Part 2** walks through the practical pipeline of building the
  training datasets each phase actually needs.

---

## The Big Picture First

All these terms are **nested inside each other**, like Russian nesting
dolls. Here's the relationship before we unpack each one:

```
┌───────────────────────────────────────────────────────────────┐
│  DIGITAL SIGNAL PROCESSING (DSP)                                │
│  "General techniques for working with digital sound"            │
│                                                                    │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │  AUDIO MACHINE LEARNING                                    │  │
│   │  "DSP feeding into learned models: classify, recognize,    │  │
│   │   understand sound"                                        │  │
│   │                                                              │  │
│   │    ┌───────────────────────────────────────────────────┐  │  │
│   │    │  GENERATIVE AUDIO AI                                │  │  │
│   │    │  "Models that produce NEW audio, not just analyze   │  │  │
│   │    │   existing audio"                                   │  │  │
│   │    │                                                       │  │  │
│   │    │   ┌───────────────────────────────────────────┐    │  │  │
│   │    │   │  aarambh-voice-studio                       │    │  │  │
│   │    │   │  TTS + cloning + music + singing + composing│    │  │  │
│   │    │   └───────────────────────────────────────────┘    │  │  │
│   │    └───────────────────────────────────────────────────┘  │  │
│   └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

Keep this picture in mind — everything below fits into one of these
nested boxes.

---

# PART 1 — Understanding the Terms

## 1. Digital Signal Processing (DSP)

**Definition:** DSP is the broad field of techniques for working with
digital representations of physical signals — sound, in this project's
case, but the same ideas apply to images, radio signals, and more.

**Beginner explanation:**
DSP is the *umbrella term* — the biggest circle in our diagram above. It
includes decades-old, well-understood techniques like filtering
(removing unwanted frequencies), resampling (changing how many numbers
represent one second of sound), and transforms like the STFT (turning a
waveform into a time-vs-frequency picture). None of this inherently
involves "learning" — a lot of DSP is fixed, hand-designed math that's
been used since long before modern AI.

**Why it matters:**
Every audio ML technique in this project sits *on top of* DSP — the
codec's STFT-based losses, the music classifier's spectrogram input, and
the mixing model's loudness calculations are all DSP operations, just
combined with learned models rather than used entirely on their own.

**Example:**
```
Old-school DSP (no learning involved):
  A low-pass filter removes all frequencies above a cutoff —
  a fixed mathematical rule, the same every time, no training needed

Modern audio ML (DSP + learning):
  Show a classifier thousands of labelled spectrograms →
  it learns to recognize "this is a saxophone" on its own,
  no human wrote an explicit rule for what a saxophone "looks like"
  in a spectrogram
```

**Diagram:**
```
   DSP = any technique for processing digital sound
        │
        ├── Fixed, hand-designed techniques (filters, resampling, STFT)
        │
        └── Audio Machine Learning (learns patterns from data) ← below
```

**Common beginner questions:**
- *Q: Is STFT (Math Guide §2) "AI"?* → No — it's a fixed mathematical
  transform, not a learned model. It becomes part of an *AI system* when
  its output (a spectrogram) is fed into a learned classifier or used in
  a learned loss function.
- *Q: Do I need to understand classical DSP deeply to work on this
  project?* → Not deeply, but the core ideas (sample rate, spectrograms,
  windowing) come up constantly — the Math Guide's early sections cover
  exactly what's needed.

---

## 2. Audio Machine Learning

**Definition:** Audio Machine Learning is DSP output (spectrograms,
waveforms) fed into models that learn patterns from labelled or unlabeled
examples, instead of following fixed hand-written rules.

**Beginner explanation:**
Instead of a programmer writing "if the spectrogram has energy at these
specific frequencies, it's a violin," you show a model thousands of
labelled audio clips, and it learns on its own which patterns tend to
indicate "violin," "genre: jazz," or "this word was said."

**Why it matters:**
This is the foundational idea behind Phase 11 (Music Understanding) —
instead of hand-coding rules for what makes something "lo-fi hip hop,"
the classifier is trained on labelled examples and learns the pattern
statistically.

**Example:**
```
Rule-based genre detector (NOT ML):
  Human writes: "if tempo is between 70-90 BPM AND has vinyl crackle
  sound, classify as lo-fi hip hop"
  → Breaks the moment a genre's conventions shift even slightly

ML-based genre detector:
  Show it 50,000 labelled tracks across many genres
  → It learns dozens of subtle patterns no human explicitly wrote down,
    and generalizes to tracks that don't match any single hand-written rule
```

**Diagram:**
```
  Labelled audio examples
         │
         ▼
  ┌─────────────┐
  │   Model       │  learns statistical patterns
  │  (training)   │
  └─────────────┘
         │
         ▼
  Can classify NEW, never-before-seen audio
```

**Common beginner questions:**
- *Q: Is ASR (speech-to-text) audio ML?* → Yes — it's audio ML applied
  specifically to recognizing spoken words, and this project uses it
  (Phase 24) purely as a *quality-checking* tool, not as a user-facing
  feature.
- *Q: Does audio ML always need huge datasets?* → Generally yes for good
  results, though techniques like fine-tuning (Phase 21) and
  self-learning (Phase 23) are specifically designed to work well with
  much smaller amounts of new data on top of an already-trained model.

---

## 3. Generative Audio AI

**Definition:** Generative Audio AI is audio ML specifically aimed at
*producing new audio*, rather than just analyzing or classifying audio
that already exists.

**Beginner explanation:**
A classifier (Phase 11) listens to music and tells you what genre it is
— it never creates anything new. A generative model (Phase 6, 12, 13)
does the opposite: given a prompt (text, a style description, lyrics), it
produces audio that didn't exist before. This is the harder, newer
problem, and it's what most of `aarambh-voice-studio` actually is.

**Why it matters:**
This distinction — understanding vs. generating — directly explains why
this project's roadmap builds Music Understanding (Phase 11) *before*
Music Generation (Phase 12): the understanding model becomes a tool that
helps build data for, and later evaluate, the generative model.

**Example:**
```
Classification (understanding): "this clip is lo-fi hip hop, 82 BPM"
       — analyzes something that already exists

Generation: "make me something that sounds like lo-fi hip hop, 82 BPM"
       — produces something brand new that never existed before
```

**Diagram:**
```
  Generative Audio AI
        │
        ├── Text-to-Speech (Phase 6)          ─┐
        ├── Voice Cloning (Phase 8)             │
        ├── Background Music Generation (12)    ├── all generate NEW audio
        ├── Singing Synthesis (13-14)            │
        └── Song Composition (17)               ─┘
```

**Common beginner questions:**
- *Q: Is Phase 11's music classifier "generative"?* → No — it's
  understanding-focused (audio ML, not generative audio AI), even though
  it's a crucial *supporting* tool for the generative Music Engine.
- *Q: Why is generation considered "harder" than understanding?* →
  Classifying correctly requires recognizing existing patterns; generating
  requires *producing* something new that's coherent, high-quality, and
  matches a prompt — many more ways to fail, and much harder to check
  automatically (which is exactly why Phase 24's evaluation harness is
  such a large part of this project).

---

## 4. Waveform & Sample Rate

**Definition:** A waveform is the raw list of numbers representing air
pressure over time — what a microphone actually records. The sample rate
is how many of those numbers represent one second of sound.

**Beginner explanation:**
If you zoomed into a recording of your voice far enough, you'd see it's
really just a very long list of numbers, each one very close to the
number before and after it, changing smoothly over time — like a
super-detailed connect-the-dots drawing of air pressure.

**Why it matters:**
Every single phase of this project ultimately traces back to this raw
representation — the codec (Phase 1) exists specifically because this
raw form is too large and unwieldy for a transformer to work with
directly.

**Example:**
```
0.5 seconds of audio at 24kHz = 12,000 numbers

[0.0, 0.02, 0.05, 0.03, -0.01, -0.04, -0.02, 0.01, ...]

Each number is about 0.042 milliseconds apart from the next.
```

**Diagram:**
```
  microphone
      │
      ▼  (samples air pressure thousands of times per second)
  [0.0, 0.02, 0.05, 0.03, -0.01, ...]  ← the waveform
```

**Common beginner questions:**
- *Q: Why 24kHz for speech and 44.1kHz for music in this project?* →
  Speech's most important frequencies (for intelligibility) fit
  comfortably under 24kHz's range; music, especially with cymbals and
  other bright, high-frequency instruments, benefits from the extra
  detail 44.1kHz captures.
- *Q: Can a waveform be negative?* → Yes — air pressure oscillates above
  and below a resting point, so waveform values are typically centered
  around 0, ranging roughly from -1.0 to 1.0 once normalized.

---

## 5. Spectrogram & Mel Spectrogram

**Definition:** A spectrogram is a picture of sound — time along one
axis, frequency along the other, brightness showing how strong each
frequency is at each moment. The mel spectrogram is the same idea,
re-scaled to match human hearing.

**Beginner explanation:**
A raw waveform is hard to "see" patterns in — it just looks like a
squiggly line. A spectrogram reorganizes the same information into
something that looks more like a picture, where patterns like "this is a
vowel sound" or "this is a drum hit" become visible shapes, the same way
a sheet of music makes patterns visible that would be much harder to spot
in a raw audio recording.

**Why it matters:**
Phase 1's codec discriminators and Phase 11's music classifier both work
on spectrograms (or mel spectrograms), not raw waveforms directly —
because the patterns they need to recognize are much easier to learn in
this picture-like form.

**Example:**
```
Raw waveform: [0.02, -0.01, 0.05, ...]  ← hard to "see" any structure

Spectrogram: a 2D grid, time across, frequency up/down, brightness =
  strength — a sustained note shows up as a bright horizontal line at
  its pitch's frequency, held steady over time; a drum hit shows up as
  a bright vertical streak across many frequencies at one instant
```

**Diagram:**
```
  waveform ──► STFT (Math Guide §2) ──► linear spectrogram
                                              │
                                              ▼ re-bin to mel scale
                                       mel spectrogram
                                       (matches human hearing)
```

**Common beginner questions:**
- *Q: Do I ever see a spectrogram directly as a user of this project?* →
  Not typically — it's an internal representation used during training
  and by internal classifiers/discriminators, not something exposed in
  the CLI or API.
- *Q: Is a mel spectrogram always better than a plain one?* → For tasks
  involving human perception of sound (classification, quality
  discriminators), yes generally — but the plain, linear spectrogram is
  still useful in places where exact frequency detail matters more than
  perceptual matching.

---

## 6. Neural Audio Codec & Audio Tokens

**Definition:** A neural audio codec is a learned system that compresses
a waveform down into a short list of discrete tokens, and can reconstruct
a close copy of the original waveform from those tokens.

**Beginner explanation:**
Think of it as a specialized, learned "zip and unzip" for sound — except
instead of being designed by an engineer with fixed rules (like an MP3
encoder), the compression scheme itself is learned from data, and instead
of aiming purely for "sounds good to a human," part of it is specifically
trained to produce tokens that a *transformer* can easily learn to
predict.

**Why it matters:**
Every generative engine in this project (TTS, music, singing) generates
audio by predicting the next token from this codec's vocabulary — nothing
downstream is possible without it. It's this project's equivalent of
`aarambh-studio`'s text tokenizer.

**Example:**
```
waveform: [0.02, -0.01, 0.05, ...]  (24,000 numbers, 1 second)
                │
                ▼  Codec
tokens: [451, 12, 998, 12, 87, ...]  (~13 numbers, 1 second)
```

**Diagram:**
```
  waveform ──► Encoder ──► RVQ tokens ──► Decoder ──► reconstructed waveform
                 (Phase 1, frozen after training — see Complete Guide)
```

**Common beginner questions:**
- *Q: Is this the same as an MP3 or a WAV file?* → No — MP3 is a
  human-designed, fixed compression scheme aimed at sounding good to a
  listener. This codec is a *learned* model whose tokens are specifically
  shaped to be easy for a transformer to predict, which is a different
  goal even though both involve "compression."
- *Q: Why not just use MP3's compression directly instead of building a
  new codec?* → MP3's compressed representation isn't structured as a
  fixed vocabulary of discrete "words" a transformer could predict one at
  a time — a purpose-built neural codec is needed for the
  next-token-prediction approach this project uses throughout.

---

## 7. TTS (Text-to-Speech)

**Definition:** TTS is the task of converting written text into spoken
audio.

**Beginner explanation:**
This is the most classic audio generation task — you type a sentence, the
system reads it aloud in a synthetic voice. `aarambh-voice-studio`'s
Voice Engine (Phase 6 onward) is built around this task, with cloning,
voice design, and emotion control all layered on top of the same core
mechanism.

**Why it matters:**
TTS is the foundation everything else in the Voice Engine builds on —
cloning (Phase 8) and emotion control (Phase 10) are both TTS with extra
conditioning added, not separate systems built from scratch.

**Example:**
```
aarambh-voice-studio speak --text "Hello from Aarambh Voice Studio" --out hello.wav

  → produces a WAV file containing synthesized speech of that sentence
```

**Diagram:**
```
  "Hello from Aarambh Voice Studio"
          │
          ▼ Phase 2 (text prep) → Phase 6 (TTS model) → Phase 1 (codec decode)
          ▼
  hello.wav
```

**Common beginner questions:**
- *Q: Is TTS the same as voice cloning?* → No — plain TTS uses a
  preset/default voice; voice cloning (Phase 8) specifically conditions
  the same TTS mechanism on a reference speaker's embedding to sound like
  a particular person.
- *Q: Does TTS quality depend more on the codec or the transformer?* →
  Both matter, but a flawed codec (Phase 1) caps the *ceiling* of possible
  quality no matter how good the transformer gets — which is exactly why
  the codec is built, frozen, and quality-checked first.

---

## 8. ASR (Automatic Speech Recognition) & WER (Word Error Rate)

**Definition:** ASR converts spoken audio into written text — the reverse
of TTS. WER measures what fraction of words an ASR system (or a TTS
system's output, checked via ASR) got wrong.

**Beginner explanation:**
In this project, ASR isn't a user-facing feature — it's used purely as a
*quality check*. Generate speech from some text, run it back through an
ASR system, and see if the transcription matches the original text. If it
doesn't match well, the generated speech probably wasn't very
intelligible.

**Why it matters:**
This "round-trip" check (Phase 24) is one of the most important
automatic quality signals in the whole project — it directly catches
unintelligible or garbled generated speech without needing a human
listener.

**Example:**
```
Original text:        "the quick brown fox jumps over the lazy dog"
Generated speech → ASR transcription: "the quick brown fox jumps over the lazy dog"
  → 0% WER, perfect intelligibility

Original text:        "the quick brown fox jumps over the lazy dog"
Generated speech → ASR transcription: "the quick brown fox jump over lazy dog"
  → some words missing/wrong → non-zero WER, a real quality problem
```

**Diagram:**
```
  original text
        │
        ▼ Phase 6 TTS model
  generated speech
        │
        ▼ ASR (round-trip check)
  transcribed text
        │
        ▼ compare against original
  WER score (lower = more intelligible)
```

**Common beginner questions:**
- *Q: Does this project ship its own ASR system?* → No — Phase 24's
  evaluation harness uses ASR purely as a measurement tool, not as a
  user-facing product feature; a separate, already-existing ASR
  capability is what's leveraged here.
- *Q: Can WER be 0% and the speech still sound bad?* → Yes — WER only
  measures intelligibility (were the right words understood), not
  naturalness. That's exactly why MOS-proxy (§11 below) exists as a
  separate metric.

---

## 9. Phoneme & G2P (Grapheme-to-Phoneme)

**Definition:** A phoneme is a single distinct unit of sound in a
language. G2P is the process of converting written letters (graphemes)
into the phonemes they actually represent when spoken.

**Beginner explanation:**
English spelling doesn't reliably tell you how to pronounce a word — the
letters "ough" sound completely different in "though," "through," and
"cough." G2P is the step that resolves this ambiguity, converting
spelling into the actual sounds a mouth would make, before those sounds
get generated as audio.

**Why it matters:**
Without G2P (Phase 2), the TTS model would have to guess pronunciation
purely from spelling patterns learned during training — workable much of
the time, but unreliable for names, borrowed words, and homographs.

**Example:**
```
Written word: "read" (present tense) → phonemes for "reed"
Written word: "read" (past tense)    → phonemes for "red"

Same spelling, different phonemes — context determines which one G2P
should pick.
```

**Diagram:**
```
  written text ──► G2P ──► phoneme sequence ──► fed into TTS model
```

**Common beginner questions:**
- *Q: Is G2P the same across every language?* → No — this project's G2P
  (Phase 2) has to handle English, Hindi, and Sanskrit, and even
  code-switching within a single sentence, each with different
  pronunciation rules.
- *Q: What if a word isn't in the pronunciation dictionary at all?* → A
  small learned fallback model predicts likely phonemes directly from the
  spelling, the same way a human sounds out an unfamiliar word.

---

## 10. Zero-Shot Voice Cloning & Speaker Embedding

**Definition:** Zero-shot voice cloning is making a model speak in a new
voice using only a short reference sample, with no additional training. A
speaker embedding is the fixed-length list of numbers that represents a
specific voice's identity.

**Beginner explanation:**
"Zero-shot" means the model has never specifically trained on this exact
person's voice — it's generalizing from thousands of *other* voices seen
during training, the way a person can often mimic a new accent
reasonably well after hearing just a few sentences of it.

**Why it matters:**
This is the mechanism (Phase 8) that makes `aarambh-voice-studio clone`
possible without retraining anything for each new voice — a real
practical necessity for any usable product.

**Example:**
```
8-second reference clip
      │
      ▼ Speaker encoder
speaker_embedding = [0.12, -0.45, 0.88, ...]   (256 numbers)
      │
      ▼ fed into the TTS model alongside new text
new speech, in that person's voice, saying something they never said
```

**Diagram:**
```
  reference audio ──► Speaker encoder ──► speaker_embedding (256 numbers)
                                                  │
                                                  ▼ injected at layer 0
                                          Transformer core (Phase 4)
```

**Common beginner questions:**
- *Q: How short can the reference clip be?* → This project targets 3-10
  seconds — long enough to capture distinguishing voice characteristics,
  short enough to be practical to collect.
- *Q: Does the model "remember" a cloned voice for later?* → Not by
  default — a zero-shot clone only uses the embedding for that one
  request. Making the system genuinely *remember* a voice for future
  requests is what Phase 23's self-learning system is specifically for.

---

## 11. MOS & MOS-Proxy (Mean Opinion Score)

**Definition:** MOS is the traditional 1-5 scale humans use to rate audio
quality/naturalness. MOS-proxy is a learned model trained to *predict*
what MOS score humans would likely give, so quality can be measured
automatically.

**Beginner explanation:**
Getting real humans to rate every single training checkpoint's audio
quality doesn't scale — it's slow and expensive. A MOS-proxy model is
trained once, on audio that *does* have real human MOS ratings, and then
used to automatically estimate scores for new audio going forward.

**Why it matters:**
Phase 24's evaluation harness relies on MOS-proxy as its main
naturalness metric — without it, judging "does this sound more natural
now" would require human listening sessions for every checkpoint, which
isn't practical during active development.

**Example:**
```
Human-rated training set: 5,000 clips, each rated 1-5 by multiple listeners
        │
        ▼ train a small model to predict these ratings from audio features
MOS-proxy model
        │
        ▼ applied to a brand-new, never-human-rated generated clip
predicted MOS: 3.9   (an automatic estimate, no human needed for this clip)
```

**Diagram:**
```
  human-rated audio (training data for the proxy)
          │
          ▼
  ┌───────────────┐
  │  MOS-proxy      │  learns to predict human ratings
  │  model          │
  └───────────────┘
          │
          ▼
  new generated audio ──► predicted MOS score (automatic, fast)
```

**Common beginner questions:**
- *Q: Is MOS-proxy perfectly accurate compared to real humans?* → No —
  it's a useful approximation, not a perfect substitute. It's
  significantly more practical than collecting human ratings for every
  checkpoint, which is the actual tradeoff being made.
- *Q: Does this project ever use real human ratings at all?* → The
  MOS-proxy model itself needs real human ratings to be *trained* in the
  first place — after that, it's used as an automatic stand-in for
  ongoing development.

---

## 12. Music Tags: BPM, Key, Genre, Mood

**Definition:** These are the standard descriptive labels used to
categorize a piece of music — BPM (tempo, beats per minute), key (which
musical scale it's built around), genre, and mood.

**Beginner explanation:**
These are the exact outputs Phase 11's Music Understanding classifier
produces, and the exact inputs Phase 12's Music Generation model is
conditioned on — understanding and generation share this same
vocabulary of descriptive tags.

**Why it matters:**
Without a consistent tagging vocabulary, there'd be no shared language
between "what the classifier says a clip sounds like" and "what a user
asks the generator to produce" — these tags are what let Phase 11's
auto-labelling actually train Phase 12's generator.

**Example:**
```
music generate --prompt "lo-fi hip-hop, rainy, 80 bpm"
                          │       │        │
                       genre    mood     tempo
```

**Diagram:**
```
  music audio ──► classifier ──► {genre, tempo, key, mood, instruments}
                                              │
                                              ▼ same vocabulary
  text prompt ──► generator ──► music matching those same tags
```

**Common beginner questions:**
- *Q: Are these tags standardized across the whole music industry?* →
  Broadly yes for things like BPM and key (well-defined musically), less
  so for genre and mood, which can be somewhat subjective — the project's
  classifier learns whatever labelling conventions its training data
  uses.
- *Q: Can a track have more than one genre or mood tag?* → Yes — mood and
  instrumentation are multi-label (a track can be both "relaxed" and
  "nostalgic" at once), while genre and key are typically single-label
  per track in this project's design.

---

## 13. F0 (Fundamental Frequency) & MIDI

**Definition:** F0 is the base pitch of a voice or instrument at a given
moment, measured in Hz. MIDI is a standard format for representing
musical notes (pitch, timing, duration) without any actual audio.

**Beginner explanation:**
F0 is "how high or low" something sounds at any instant — central to
singing, since hitting the correct F0 at the correct time is what makes
something sound "in tune." MIDI is how this project represents the
*target* melody a singing request should follow — a list of notes and
durations, with no audio in it at all, similar to sheet music but in a
format software can read directly.

**Why it matters:**
Phase 13's Singing Engine is conditioned directly on a MIDI-like melody
input, and trained with an explicit F0 loss (Math Guide §13) so the
generated singing actually hits the notes specified, rather than just
singing the lyrics at whatever pitch feels natural.

**Example:**
```
MIDI-like melody input for "happy birthday":
  note: C4, duration: 0.5s
  note: C4, duration: 0.5s
  note: D4, duration: 1.0s
  note: C4, duration: 1.0s
  ...

Singing Engine generates audio whose F0 contour tracks these exact
notes and durations.
```

**Diagram:**
```
  MIDI melody (notes + durations)
          │
          ▼ cross-attention conditioning (Phase 4's mechanism)
  ┌───────────────┐
  │ Singing Engine  │  predicts codec tokens whose F0 matches the melody
  └───────────────┘
```

**Common beginner questions:**
- *Q: Does this project need actual .mid files, or something simpler?* →
  The architecture describes a "MIDI-like" representation — the exact
  file format is an implementation detail, but the concept (notes +
  durations, no audio) is the same as standard MIDI.
- *Q: Can F0 be measured directly from a real singer's recording?* →
  Yes — F0 extraction from real audio is exactly how ground-truth pitch
  targets are obtained for training data (see Part 2 below).

---

## 14. LUFS & Mixing

**Definition:** LUFS is a standardized loudness measurement that accounts
for how humans actually perceive loudness across different frequencies.
Mixing is the process of combining separately-generated audio tracks
(like vocals and instrumentals) into one balanced, properly-mastered
result.

**Beginner explanation:**
Two tracks can look similarly "loud" on a simple amplitude meter but
sound very differently loud to a human ear, depending on which
frequencies dominate. LUFS accounts for this. Phase 15's mixing model
uses LUFS to make sure a finished song matches a sensible loudness target
rather than sounding too quiet or blown-out compared to typical music.

**Why it matters:**
Without LUFS-aware mixing, a generated song's vocal and instrumental
could easily end up unbalanced (one drowning out the other), or the whole
song could sound noticeably quieter or louder than typical streamed
music.

**Example:**
```
Vocal stem alone:        -20 LUFS (quiet)
Instrumental stem alone: -12 LUFS (loud)

Naive combination: instrumental buries the vocal

Mixed + LUFS-matched to -14 target: vocal boosted, instrumental
  slightly reduced, both brought toward a balanced -14 LUFS overall
```

**Diagram:**
```
  vocal stem      instrumental stem
       │                  │
       └────────┬─────────┘
                 ▼
       ┌─────────────────┐
       │  Mixing network   │  balances gain, matches target LUFS
       └─────────────────┘
                 ▼
         finished mixed song
```

**Common beginner questions:**
- *Q: Is LUFS the same thing as "volume"?* → Related but more
  sophisticated — a simple volume/amplitude measurement doesn't account
  for frequency-dependent human perception the way LUFS does.
- *Q: What if I want the vocal louder than a "balanced" mix would put
  it?* → The Full Control Layer (Phase 18) exposes manual vocal/music
  gain controls specifically for cases where the default balanced mix
  isn't what's wanted.

---

## 15. Autoregressive Generation

**Definition:** Generating output one token at a time, with each new
token conditioned on everything generated so far.

**Beginner explanation:**
This is the same fundamental approach `aarambh-studio` uses for text — predict
the next piece, add it to what's already been generated, then predict the
next piece after *that*, repeating until done. This project applies the
exact same idea to audio tokens instead of word tokens.

**Why it matters:**
TTS (Phase 6), music generation (Phase 12), and Singing Stage A (Phase
13) are all autoregressive — it's the default generation strategy
throughout this project, with diffusion (§16 below) used only as a
deliberate, narrow exception for singing refinement.

**Example:**
```
Generate token 1 → [451]
Generate token 2, given token 1 → [451, 12]
Generate token 3, given tokens 1-2 → [451, 12, 998]
... continues until the desired audio length is reached
```

**Diagram:**
```
  [ ] ──► predict ──► [451]
  [451] ──► predict ──► [451, 12]
  [451, 12] ──► predict ──► [451, 12, 998]
       (each step conditions on everything generated so far)
```

**Common beginner questions:**
- *Q: Why generate one token at a time instead of all at once?* → Because
  each token's most likely value genuinely depends on what came before it
  — generating "all at once" would ignore that dependency and tend to
  produce less coherent results, though it can be faster (this is part of
  why diffusion, §16, exists as an alternative approach for some tasks).
- *Q: Does one-token-at-a-time generation mean it's slow?* → It can be,
  which is exactly why speculative decoding (Complete Guide, Phase 25)
  exists — to speed up autoregressive generation without changing its
  fundamental one-token-at-a-time logic.

---

## 16. Diffusion & Flow-Matching

**Definition:** Diffusion is a generative approach that starts from
random noise and gradually removes it in many small steps to reveal a
clean output. Flow-matching is a closely related, often faster-training
alternative.

**Beginner explanation:**
Instead of building output piece by piece like autoregressive generation
does, diffusion starts with something that looks like static/noise and
repeatedly asks "what noise should I remove here?", getting a little
closer to a clean result with every step, with the freedom to reconsider
the whole clip at each step rather than committing to one piece forever.

**Why it matters:**
This is the mechanism behind Phase 14's optional singing refinement pass
— the one place in this whole project where the "predict tokens one at a
time" philosophy is deliberately supplemented rather than used
exclusively, because it tends to produce more natural-sounding results
for singing specifically.

**Example:**
```
Start: pure random noise
Step 1: "what noise should I remove, given Stage A's output as
         conditioning?" → remove a bit
Step 2: repeat, removing a bit more noise each time
...
Final step: clean, refined sung audio emerges
```

**Diagram:**
```
  random noise
       │
       ▼ step 1 (small denoise, conditioned on Stage A's output)
       ▼ step 2
       ▼ ... many steps
       ▼
  clean, refined audio
```

**Common beginner questions:**
- *Q: Why not use diffusion everywhere instead of autoregressive
  generation?* → Diffusion typically requires many steps at generation
  time, which tends to be slower than a single autoregressive pass — this
  project uses diffusion narrowly (singing refinement only) where the
  naturalness gain is judged worth that extra cost.
- *Q: Is flow-matching just a fancier name for diffusion?* → They're
  closely related but distinct techniques — flow-matching learns a more
  direct path from noise to data, which often trains faster in practice;
  either is acceptable per this project's architecture for the singing
  refinement pass.

---

## 17. GRPO & DPO (Alignment)

**Definition:** GRPO (Group Relative Policy Optimization) and DPO (Direct
Preference Optimization) are both training methods that push a model
toward outputs that score well on quality metrics, rather than just
matching training examples exactly.

**Beginner explanation:**
Ordinary training (Phases 6, 12, 13) teaches a model to predict tokens
that match real recordings — a proxy for quality, not quality itself.
GRPO and DPO add a second training stage that directly optimizes for
metrics you actually care about (naturalness, speaker fidelity, emotion
accuracy), using the evaluation harness's own scores as the training
signal.

**Why it matters:**
This is Phase 22's mechanism for closing the gap between "predicts
training data well" and "actually sounds good" — a gap that ordinary
supervised training alone can leave on the table.

**Example:**
```
GRPO: generate 6 candidates for one prompt, score each, reinforce the
       above-average ones, discourage the below-average ones

DPO: from those same 6 scored candidates, pick the best and worst as
      a (chosen, rejected) pair, train the model to prefer the chosen one
```

**Diagram:**
```
  same prompt, sampled K times
          │
          ▼
  score each candidate with eval-harness metrics
          │
   GRPO: reinforce relative to group average
   DPO:  train on (best, worst) pairs
          │
          ▼
  model shifted toward higher-scoring generations
```

**Common beginner questions:**
- *Q: Do you need both GRPO and DPO, or just one?* → This project uses
  both — GRPO is more thorough (live sampling every step) but more
  expensive; DPO is cheaper (built from pre-sampled, pre-scored pairs) and
  used as a complementary, lower-cost pass.
- *Q: Could this alignment step make the model "cheat" the metric instead
  of genuinely improving?* → This is a real risk in reinforcement
  learning generally (called reward hacking) — which is why alignment
  changes are checked against guardrail metrics (WER, speaker-similarity)
  that shouldn't regress, not just the one metric being directly
  optimized.

---

## 18. LoRA, QLoRA, DoRA (Fine-Tuning Family)

**Definition:** These are efficient fine-tuning methods that adapt an
already-trained model to a new task by adding a small number of new
trainable parameters, instead of retraining the whole model.

**Beginner explanation:**
Retraining an entire model from scratch for every new speaker or style
would be extremely expensive and slow. LoRA-family methods instead freeze
almost all of the original model and add a small, separately-trained
"patch" of new parameters that specializes it — much cheaper, much
faster, and often nearly as effective as full retraining.

**Why it matters:**
Phase 21's deliberate fine-tuning (for a speaker you'll use extensively)
and the self-learning mechanism (Phase 23) both build on this family of
techniques, applied to voice/style adaptation instead of the text tasks
`aarambh-studio` uses them for.

**Example:**
```
Base model: 55 million parameters, fully frozen
LoRA adapter for one speaker: a few hundred thousand new parameters,
  the only part that gets trained

Result: a specialized "patch" file, much smaller than the full model,
  that noticeably improves quality for this one speaker when loaded
  alongside the frozen base
```

**Diagram:**
```
  frozen base model (unchanged)
          │
          ▼ + small trainable LoRA adapter
  combined model, specialized for a specific speaker/style
```

**Common beginner questions:**
- *Q: What's actually different between LoRA, QLoRA, and DoRA?* → QLoRA
  applies the same LoRA idea on top of a quantized (memory-shrunk) base
  model, saving even more memory; DoRA is a refinement of LoRA's
  mathematical approach that tends to match full fine-tuning quality more
  closely, at a similar cost to plain LoRA.
- *Q: Is this the same mechanism as self-learning (Phase 23)?* → Related
  family of ideas (both add small adapters rather than retraining
  everything), but self-learning adds its own extra safety mechanisms
  (gradient orthogonalization, confidence gating) specifically because it
  runs automatically, without a human reviewing the update first.

---

## 19. Self-Learning & Catastrophic Forgetting

**Definition:** Self-learning is letting a deployed system adapt from new
examples automatically, without a full offline retraining job.
Catastrophic forgetting is the failure mode this has to guard against:
learning something new accidentally erasing something the model already
knew.

**Beginner explanation:**
Imagine learning a new coworker's name and, in the process, somehow
forgetting the name of someone you've known for years — that's roughly
what catastrophic forgetting looks like in a naively-updated model. Phase
23's self-learning system is specifically designed with a mathematical
technique (gradient orthogonalization, Math Guide §18) to prevent this,
so learning voice #51 can't damage voices #1 through #50.

**Why it matters:**
Without a solution to catastrophic forgetting, any system that "learns
after deployment" would be too risky to run unattended — every new update
could silently be degrading things learned earlier, with no easy way to
notice until a user complains.

**Example:**
```
Naive online learning (no protection):
  Learn voice #51 → oops, voice #23's quality quietly got worse too

This project's self-learning (with gradient orthogonalization):
  Learn voice #51 → mathematically guaranteed not to touch the
  directions voices #1-50 rely on → voice #23 unaffected
```

**Diagram:**
```
  new sample arrives
          │
          ▼
  compute update, orthogonalize it against existing adapters
          │
          ▼
  confidence gate: does this still score well on the eval harness?
          │
   yes ──► commit           no ──► discard, nothing changes
```

**Common beginner questions:**
- *Q: Is catastrophic forgetting unique to audio models?* → No — it's a
  general problem in any machine learning system that keeps learning
  after initial training; this project's specific solution (gradient
  orthogonalization) is directly borrowed from the same anti-forgetting
  approach used in the related Manas project.
- *Q: How do you know the anti-forgetting mechanism actually works, not
  just in theory?* → Phase 23's tests include a specific regression
  check: run 50 sequential updates for 50 different speakers, then
  re-verify speaker #1's quality afterward — a concrete, testable
  guarantee rather than an assumption.

---

## 20. Consent Gating & Watermarking

**Definition:** Consent gating is requiring explicit permission before a
real person's voice can be cloned. Watermarking is embedding an inaudible
signal into generated audio so it can later be identified as
AI-generated.

**Beginner explanation:**
These are the two core safety mechanisms (Phase 19) built directly into
this project's architecture rather than added as an afterthought. Consent
gating stops cloning from happening at all without explicit permission;
watermarking makes generated audio traceable after the fact, even if
someone shares it without any label saying it's AI-generated.

**Why it matters:**
Voice cloning is exactly the kind of capability that can be misused for
impersonation or fraud if built without safeguards — these two mechanisms
are what make the difference between "a tool that can clone voices" and
"a tool that can clone voices responsibly."

**Example:**
```
Cloning request, no consent token attached → REJECTED, no audio produced

Cloning request, valid consent token attached → audio produced,
  with an inaudible watermark embedded — a detector can later confirm
  "this was generated by this system," even without any visible label
```

**Diagram:**
```
  cloning request
        │
        ▼
  consent check ──── missing/invalid ────► REJECTED
        │ valid
        ▼
  generate + embed watermark
        │
        ▼
  final audio (traceable, even without a visible label)
```

**Common beginner questions:**
- *Q: Can the watermark be removed?* → It's designed to be robust to
  normal processing like mixing and mastering (tested specifically in
  Phase 19), though no watermarking scheme is unbreakable against a
  determined adversary — it's a meaningful deterrent and traceability
  mechanism, not an absolute guarantee.
- *Q: Does consent gating apply to the self-learning system too?* → Yes —
  submitting a sample for online learning (Phase 23) requires the same
  consent gating as any other cloning-adjacent request; there's no
  separate, weaker path.

---

# PART 2 — Building the Training Datasets

## Overview: the general cleaning pipeline

Before looking at where data comes from for each phase, here's the
general shape every raw audio file goes through before it's ready for
training:

```
  raw audio file (any source)
          │
          ▼
  ┌─────────────┐
  │  Validate     │  (does it exist, is it readable, what sample rate?)
  └─────────────┘
          │
          ▼
  ┌─────────────┐
  │  Resample     │  (to the codec's target rate — 24kHz speech, 44.1kHz music)
  └─────────────┘
          │
          ▼
  ┌─────────────┐
  │  Mono/Stereo  │  (convert as appropriate for the domain)
  └─────────────┘
          │
          ▼
  ┌─────────────┐
  │  Trim silence │  (remove dead air at start/end)
  └─────────────┘
          │
          ▼
  ┌─────────────┐
  │  Forced       │  (speech/singing only — which frames = which phoneme)
  │  alignment    │
  └─────────────┘
          │
          ▼
  ┌─────────────┐
  │  Auto-label   │  (music only — run Phase 11's classifier)
  └─────────────┘
          │
          ▼
  Ready for the Data Pipeline (Phase 3) → Training Loop (Phase 6, 12, 13...)
```

---

## 21. Where to Collect Speech Data From

### 21a. Single-speaker corpora (LJSpeech-style)

**Definition:** A dataset of many hours of recordings from one speaker,
usually clean studio-quality audiobook-style readings.

**Beginner explanation:**
This kind of corpus prioritizes *depth* — lots of recordings from one
voice — over *breadth* across many voices. That's exactly the right shape
for Phase 6's first TTS training run, where the goal is proving the whole
pipeline can learn to speak intelligibly at all, not yet generalizing
across different speakers.

**Why it matters:**
Depth matters more than breadth here because with only one speaker to
learn, the model's whole capacity goes toward learning to speak clearly,
rather than also needing to learn to distinguish between speakers.

**Example:**
```
LJSpeech-style corpus: 1 speaker, ~24 hours, clean audiobook recordings

Used for: Phase 6's very first "does the whole pipeline actually work"
TTS training run
```

**Diagram:**
```
  single-speaker corpus (many hours, one voice)
          │
          ▼
  Phase 6: TTS baseline training
```

**Common beginner questions:**
- *Q: Why not start directly with a multi-speaker corpus?* → Starting
  with one speaker isolates "does the pipeline work at all" from "does
  it generalize across speakers" — two different questions best answered
  one at a time.
- *Q: Is a single-speaker corpus useful for anything beyond Phase 6?* →
  It remains a useful benchmark/sanity-check dataset throughout the
  project, even after multi-speaker corpora are introduced for cloning.

### 21b. Multi-speaker corpora (VCTK-style)

**Definition:** A dataset with many speakers (often 100+), each
contributing a few minutes of recordings.

**Beginner explanation:**
This kind of corpus prioritizes *breadth* — many different voices — over
depth per speaker. That's exactly the right shape for Phase 8's zero-shot
cloning, where the speaker encoder needs to learn what generally
distinguishes one voice from another across a wide range of people, not
just get very good at one specific voice.

**Why it matters:**
The GE2E loss (Math Guide §11) that trains the speaker encoder needs
multiple speakers *within the same training batch* to work at all — it's
a relative, contrastive loss (pulling same-speaker embeddings together,
pushing different-speaker embeddings apart), so it fundamentally requires
breadth.

**Example:**
```
VCTK-style corpus: 100+ speakers, a few minutes each

Used for: Phase 8's speaker encoder training (GE2E loss)
```

**Diagram:**
```
  many speakers, few minutes each
          │
          ▼
  Phase 8: speaker encoder training (GE2E loss needs multiple
           speakers per batch to contrast against each other)
```

**Common beginner questions:**
- *Q: Could you train cloning on a single-speaker corpus instead?* → No —
  there'd be nothing to contrast against; GE2E specifically needs
  multiple speakers in the same batch to learn what makes voices
  different from each other.
- *Q: How many speakers is "enough"?* → More is generally better for
  generalization, but VCTK-scale (100+) is a reasonable, commonly-used
  starting point that this project's roadmap targets for the initial
  cloning milestone.

---

## 22. Consent — The Part That's Different From Text Data

**Definition:** Explicit, documented permission from a real person for
their voice recording to be used in training a generative model.

**Beginner explanation:**
Voice data carries an obligation text data doesn't: the recording *is*
someone's identity, in a way a paragraph of scraped web text simply
isn't. A dataset's general research license doesn't automatically cover
"can this specific voice be used to train a model that can later clone
voices" — that's a separate, more specific question.

**Why it matters:**
This project's Code of Conduct and Contributing guide explicitly forbid
adding real voice recordings (your own or anyone else's) as example
fixtures without documented consent — even for something as small as a
unit test.

**Example:**
```
A public speech corpus's license says:
  "free to use for academic research and model training"
  → this DOES cover training a TTS/cloning model (check the exact wording
    every time, but this phrasing is a reasonable green light)

A different corpus's license says:
  "free to use for transcription benchmarking only"
  → this does NOT clearly cover generative model training — do not use
    it for that purpose without checking with the source directly
```

**Diagram:**
```
  candidate speech corpus
          │
          ▼
  read the license terms specifically for "generative model training,"
  not just "research use" in general
          │
   covers it? ──► use, with source/license recorded in the manifest
          │
   unclear/no? ──► do not use for this project
```

**Common beginner questions:**
- *Q: Does this apply to the self-learning system too?* → Yes — Phase
  23's consent requirement applies at the moment a sample is submitted
  for online learning, not just at the original dataset-building stage.
- *Q: What about my own voice, for quick local testing?* → Even your own
  voice recordings shouldn't be added as committed example fixtures in
  the repository per `CONTRIBUTING.md` — use synthetic tones or clearly
  public-domain clips for that purpose instead.

---

## 23. Where to Collect Music Data From

**Definition:** Publicly available, appropriately licensed collections of
tagged or untagged music tracks.

**Beginner explanation:**
Similar to speech, music data comes in different useful shapes: smaller,
human-tagged corpora (genre, mood labels already provided) for training
Phase 11's classifier, larger untagged corpora that Phase 11's classifier
can then auto-label at scale for Phase 12's generator, and multitrack
corpora with isolated stems for Phase 15's mixing model.

**Why it matters:**
Phase 12's music generator needs a lot more (audio, style_prompt) training
pairs than any human could realistically hand-label — this is the entire
reason Phase 11's auto-labelling classifier exists before Phase 12 in the
roadmap.

**Example:**
```
FMA/MTG-Jamendo-style corpus: genre/mood/tempo-tagged tracks
  → trains Phase 11's classifier directly

Larger, untagged (but licensed) music corpus
  → auto-labelled by Phase 11's trained classifier
  → becomes Phase 12's large-scale training data

MUSDB18-style corpus: isolated stems (vocal, drums, bass, other)
  → trains Phase 15's mixing model, which needs (stems → reference mix)
    pairs specifically, not just a finished mix
```

**Diagram:**
```
  small, human-tagged corpus ──► trains Phase 11's classifier
                                          │
                                          ▼
  large, untagged corpus ──► auto-labelled ──► Phase 12's training data

  multitrack stem corpus ──► trains Phase 15's mixing model directly
```

**Common beginner questions:**
- *Q: Is auto-labelled data as reliable as human-labelled data?* → Not
  perfectly — it inherits whatever blind spots exist in the smaller
  human-labelled set the classifier learned from. This project's
  recommendation is to spot-check a sample of auto-labels against human
  judgment before trusting them at scale.
- *Q: Why can't the mixing model (Phase 15) just use the same tagged
  corpus as the classifier?* → Mixing needs *isolated stems* specifically
  (separate vocal/drum/bass tracks plus a reference finished mix) — a
  differently-shaped dataset than tagged-but-already-mixed tracks.

---

## 24. Where to Collect Singing Data From

**Definition:** Recordings of sung vocals with corresponding lyrics and
melody (MIDI-like) alignment already provided or derivable.

**Beginner explanation:**
This is the hardest data to source at scale of the three domains —
public singing corpora with proper lyric+melody alignment are less
common than speech or general music corpora, so this project expects to
combine a public reference corpus (for pipeline structure) with a smaller,
manually-annotated seed set for languages/styles not otherwise covered.

**Why it matters:**
Phase 13's Singing Engine needs (audio, lyrics, melody) triples
specifically — not just audio, and not just audio+lyrics without timing
information, since the whole point of the melody conditioning is
training the model to hit specific notes at specific times.

**Example:**
```
Public reference corpus (Opencpop-style): provides the *shape* of the
  pipeline — lyrics + MIDI already aligned, useful even if language
  coverage doesn't match your specific needs

Small internal seed set: a handful of a cappella recordings, manually
  transcribed to MIDI, with clear licensing/consent — used to bootstrap
  coverage for languages/styles the public corpus doesn't include

Once Phase 13's model is trained on the seed set, it can help speed up
alignment for additional data faster than fully-manual annotation would.
```

**Diagram:**
```
  public singing corpus (lyrics + MIDI already aligned)
          │
          ▼ establishes pipeline shape
  small manually-annotated seed set (your own languages/styles)
          │
          ▼
  Phase 13 training data
```

**Common beginner questions:**
- *Q: Why is singing data harder to get than speech or music data?* →
  Precise lyric-to-melody-to-audio alignment is a more specialized
  annotation task than simple transcription or tagging — fewer public
  corpora provide it out of the box, especially for languages outside
  major Western pop music traditions.
- *Q: Can the model help annotate its own additional training data?* →
  Yes, once trained on a seed set, using the model's own alignment
  predictions as a starting point (double-checked, not blindly trusted)
  is faster than starting every new recording's annotation completely
  from scratch.

---

## 25. Manifest / JSONL Formatting

**Definition:** A JSONL (JSON Lines) manifest is a file with one JSON
object per line, describing one training example — used so large datasets
can be streamed rather than loaded entirely into memory at once.

**Beginner explanation:**
Instead of one giant file containing every recording's data all at once
(which would need to fully load into memory before training could even
start), JSONL keeps each example on its own line, so the data loader
(Phase 3) can read and process one line at a time.

**Why it matters:**
Every dataset loader in `aarambh-voice-data` expects this exact format —
it's the common interface between "however the raw data was collected"
and "what the training loop actually consumes."

**Example (speech manifest entries):**
```jsonl
{"audio_path": "data/ljspeech/wavs/LJ001-0001.wav", "text": "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition", "speaker_id": "ljspeech_00", "license": "public-domain", "sample_rate": 22050}
{"audio_path": "data/vctk/p225/p225_001.wav", "text": "Please call Stella.", "speaker_id": "vctk_p225", "license": "vctk-1.0-open", "sample_rate": 48000}
```

**Example (music manifest entry, post-auto-labelling):**
```jsonl
{"audio_path": "data/fma/track_001.wav", "genre": "lo-fi hip hop", "tempo_bpm": 82, "key": "C minor", "mood": ["relaxed", "rainy"], "instrumentation": ["piano", "vinyl_crackle", "soft_drums"], "license": "cc-by-4.0"}
```

**Diagram:**
```
  raw collected data (varied formats, varied sources)
          │
          ▼ standardize
  one JSONL manifest, one line per example
          │
          ▼
  Phase 3's data loaders read this directly, streaming line by line
```

**Common beginner questions:**
- *Q: Why not just use a big CSV file instead?* → JSONL handles nested,
  variable-shaped data (like a music entry's multi-label `mood` field)
  much more naturally than flat CSV columns can.
- *Q: Does every manifest entry need a `license` field?* → Yes, by this
  project's convention — every entry records its source/license
  explicitly, so nothing enters training with unclear provenance.

---

## 26. Train / Validation / Test Splitting (By Speaker, Not Just By Clip)

**Definition:** Dividing a dataset into three parts — training,
validation (used during development to check progress), and test (held
out entirely, used only for final evaluation) — with audio data requiring
an extra precaution: splitting by *speaker or source*, not just by
individual utterance.

**Beginner explanation:**
If the same speaker's utterances end up in both the training and
validation sets, validation scores will look better than the model
actually generalizes — the model may just be "recognizing" a speaker
it's already memorized, rather than genuinely demonstrating zero-shot
cloning ability on a truly unseen voice.

**Why it matters:**
This is the single most common way audio ML evaluation numbers end up
misleadingly optimistic — getting this split wrong doesn't cause an
obvious crash or error, it just quietly makes every quality number look
better than reality.

**Example:**
```
WRONG split (by individual clip):
  Speaker A's clip 1, 2, 3 → training
  Speaker A's clip 4       → validation
  → validation score looks great, but the model may have just
    memorized Speaker A's voice characteristics from clips 1-3

RIGHT split (by speaker):
  Speaker A (all clips)           → training
  Speaker B (all clips)           → validation
  Speaker C (all clips, untouched)→ held out for Phase 24's final eval
  → validation score reflects genuine generalization to an unseen speaker
```

**Diagram:**
```
  full speaker/track list
          │
          ▼ split by speaker/source, never by individual clip
  ┌─────────┬─────────┬─────────┐
  │  90%      │   5%      │   5%      │
  │ training  │validation │  test     │
  └─────────┴─────────┴─────────┘
                              (never touched until Phase 24's
                               final evaluation, not during
                               training or tuning)
```

**Common beginner questions:**
- *Q: Does this rule apply to music and singing data too, or just
  speech?* → The same principle applies wherever generalization is the
  goal — splitting by track/artist for music, by singer for singing data,
  not just by individual clip.
- *Q: What's the actual danger of getting this wrong?* → You might ship a
  fine-tune or alignment change believing it improved generalization,
  when it actually only improved memorization of speakers already seen
  during training — a mistake that only becomes obvious once real users
  try genuinely new voices.

---

## 27. Licensing and Ethics Summary

**Definition:** The set of practices this project follows to make sure
every piece of training data is used with appropriate permission and
properly documented.

**Beginner explanation:**
This isn't a single technique so much as a discipline applied
consistently across every dataset entering the project — every source
checked, every license recorded, every voice's consent verified
specifically for generative-model training, not just assumed from a
general "research use" label.

**Why it matters:**
Getting this wrong isn't just a legal risk — it's the difference between
a project that treats real people's voices and creative work with
respect, and one that doesn't, which is exactly the standard this
project's Code of Conduct commits to.

**Example:**
```
Checklist applied to every new dataset before use:
  [ ] License explicitly recorded (not just "found it online")
  [ ] License explicitly covers generative-model training
      (not just general research/transcription use)
  [ ] Voice data has consent specifically for cloning/generation
  [ ] No real person's voice committed to the repo as an example fixture
  [ ] Auto-labelled data spot-checked against human judgment
```

**Diagram:**
```
  candidate dataset
          │
          ▼
  license + consent checklist (above)
          │
   passes? ──► recorded in manifest, used for training
          │
   fails?  ──► not used, regardless of how useful it would be
```

**Common beginner questions:**
- *Q: Does this slow down building the project?* → Somewhat, yes — but
  it's a deliberate tradeoff this project makes consistently, matching
  the same discipline `aarambh-studio` applies to its own text training data.
- *Q: What happens to data that turns out to have been used
  inappropriately?* → It should be removed and any models trained
  significantly on it reconsidered — this is exactly why documenting
  license/consent per entry in the manifest (§25) matters: it makes this
  kind of audit and correction possible after the fact.

---

*This completes the three-part reading path for `aarambh-voice-studio`:
`VOICE_STUDIO_COMPLETE_GUIDE_PART1/2.md` for what each phase does,
`VOICE_STUDIO_MATH_FORMULAS_GUIDE_PART1/2.md` for the exact math, and this
file for terminology and where the data comes from.*
