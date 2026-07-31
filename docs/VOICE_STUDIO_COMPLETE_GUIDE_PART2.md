# aarambh-voice-studio: The Complete Beginner's Guide (Part 2 of 2)

### Continues directly from Part 1, Phase 13

Same format as Part 1, for every phase:
- A **plain-English definition**
- A **beginner explanation**
- **Why we actually need it**
- A **real-world example**
- A **diagram**
- **Common beginner questions**

---

## Phase 14: Singing Synthesis Stage B — Diffusion Refinement

**Definition:** An optional second pass over Stage A's output that makes
the singing sound more natural, using a different generation technique
(diffusion) instead of predicting tokens one at a time.

**Beginner explanation:**
Autoregressive generation (everything in Phases 1-13) predicts one token,
then the next, then the next — each choice locked in before moving on.
Diffusion works differently: it starts from random noise and gradually
"cleans it up" over many small steps, with each step allowed to
reconsider the whole clip rather than just committing to one token
forever. For singing specifically, this second approach tends to produce
more natural-sounding results — it's the one place in this whole project
where the "predict tokens one at a time" philosophy is deliberately
supplemented rather than used exclusively.

**Why we need it:**
Autoregressive-only singing (Phase 13 alone) is fully functional but is
the part of this project most likely to sound the least natural compared
to current research elsewhere. Stage B exists specifically to close that
gap, while staying optional so Phase 13 alone still ships a complete,
working Singing Engine even if Stage B isn't enabled.

**Example:**
```
Stage A output: a sung phrase, correct pitch and timing, slightly
                 "flat"-sounding compared to a real singer

        │
        ▼ Stage B: several small denoising steps, conditioned on
          Stage A's output as a starting "sketch"
        │
        ▼
refined output: same pitch and timing, but with more natural vibrato,
                 breath, and vocal texture — rated higher by the
                 MOS-proxy naturalness metric in a side-by-side test
```

**Diagram:**
```
  Stage A output (codec tokens, correct pitch/timing)
          │
          ▼
  ┌───────────────┐
  │  Start from     │  pure random noise
  │  random noise   │
  └───────────────┘
          │
          ▼  step 1: "what noise should I remove, given Stage A's output
          ▼           as conditioning?"
          ▼  step 2: remove a little more noise
          ▼  ... (many small steps)
          ▼
  refined, more natural-sounding sung audio
```

**Common beginner questions:**
- *Q: Why not use diffusion for everything, if it sounds more natural?* →
  Diffusion is generally slower at generation time (many small steps vs.
  one autoregressive pass) and this project's whole philosophy is
  "predict tokens one at a time" for consistency and speed — Stage B is
  a deliberate, narrow exception where the naturalness gain is judged
  worth the extra cost, specifically for singing.
- *Q: What happens if you disable Stage B?* → The system falls back
  cleanly to Stage A's output alone — a fully working Singing Engine,
  just without the extra naturalness pass.

---

## Phase 15: Singing + Music Mixing

**Definition:** Combining a separately-generated vocal track and
instrumental track into one properly-balanced, mastered song.

**Beginner explanation:**
Vocals and instrumentals are generated completely independently (Phases
12-14) — this phase is where they get combined, the same way a sound
engineer mixes separately-recorded tracks in a real studio. It handles
two things: making sure the vocal is loud enough to be heard over the
instrumental (and vice versa), and making sure the final loudness matches
a broadcast standard (LUFS — see math guide) so the song doesn't sound
too quiet or too loud compared to other music people listen to.

**Why we need it:**
Simply playing two audio tracks at the same time, unmixed, usually sounds
bad — one drowns out the other, or the overall loudness is wildly
inconsistent with what listeners expect. This phase is what turns
"vocals + music" into "a song."

**Example:**
```
Vocal stem:        peak loudness -6 dB, fairly quiet overall
Instrumental stem: peak loudness -3 dB, louder than the vocal

Naive combination: instrumental buries the vocal, hard to hear the lyrics

Mixed + mastered:  vocal gain boosted, instrumental gain slightly
                   reduced, both brought to a target of -14 LUFS overall
                   → vocal sits clearly on top, instrumental supports it
```

**Diagram:**
```
  vocal stem            instrumental stem
       │                        │
       └───────────┬────────────┘
                    ▼
          ┌───────────────┐
          │  Mixing        │  balances vocal vs. instrumental gain
          │  network        │
          └───────────────┘
                    ▼
          ┌───────────────┐
          │  Loudness      │  matches target LUFS (e.g. -14)
          │  matching       │
          └───────────────┘
                    ▼
              finished mixed song
```

**Common beginner questions:**
- *Q: Why not just let the user manually set volume levels?* → The
  control layer (Phase 18) does expose manual gain controls for users who
  want them — but the default mixing network exists so a request with no
  manual tweaking still produces a well-balanced result out of the box.
- *Q: What is a "target LUFS" and why -14?* → -14 LUFS is a widely used
  streaming-platform loudness target — matching it means a generated song
  won't sound jarringly quieter or louder than other music when played
  back-to-back with it.

---

## Phase 16: Cloning + Emotion Extended to Singing

**Definition:** Making the voice cloning (Phase 8) and emotion control
(Phase 10) systems work inside the Singing Engine, not just in ordinary
speech.

**Beginner explanation:**
This phase doesn't invent new mechanisms — it proves that the mechanisms
already built for speech generalize to singing, by wiring the same
speaker-embedding and emotion-embedding conditioning into the Singing
Engine's transformer, and running a smaller additional training pass so
the model learns to apply them correctly in a sung context (which is more
demanding than speech, since pitch and duration are already fixed by the
melody, leaving less freedom to "adjust" for a speaker's natural
characteristics).

**Why we need it:**
Without this phase, cloning and emotion would only work for spoken text —
"sing this in my voice, sounding joyful" wouldn't be possible even though
both individual pieces (cloning, emotion, singing) would technically
already exist separately.

**Example:**
```
aarambh-voice-studio sing --lyrics "..." --melody melody.mid \
  --reference ref.wav --emotion "joyful" --out cloned_emotional_singing.wav

  → sung vocals matching the melody, in the reference speaker's voice,
    with prosody shaped by the "joyful" emotion embedding
```

**Diagram:**
```
  reference audio ──► speaker_embedding  ─┐
                                            ├──► Singing Engine transformer
  "joyful" ──► emotion_embedding  ────────┘         (same injection points
                                                       as Phase 4/8/10)
          lyrics + melody ──────────────────────►
                                                   │
                                                   ▼
                            sung audio: correct pitch/timing + cloned
                            voice + emotional prosody, all at once
```

**Common beginner questions:**
- *Q: If the mechanisms already exist, why does this need its own
  training pass at all?* → Applying conditioning correctly *while also*
  hitting exact melody-dictated pitch and duration is a harder combined
  task than either piece alone — a small additional training pass tunes
  the model specifically for that combination, rather than assuming it
  transfers perfectly for free.
- *Q: Does this reuse Phase 19's consent gating?* → Yes — cloning a voice
  for singing requires exactly the same consent token as cloning a voice
  for speech; there's no separate, weaker path for singing.

---

## Phase 17: Structure Planner + Song Composer

**Definition:** The system that looks at a full set of lyrics and figures
out its shape — which lines are verses, which are the chorus, whether a
chorus repeats — before generating any audio, and then orchestrates the
Voice, Music, and Singing Engines to build the complete song.

**Beginner explanation:**
Without a structure planner, a "compose a song" request would treat the
lyrics as one long flat block of text and generate it start to finish,
with no sense that the chorus should probably sound bigger, or that a
repeated chorus should ideally sound the same way each time it repeats
(real songs do this; ignoring it makes generated songs sound subtly wrong
even when every individual line sounds fine on its own). This phase adds
a small model that reads the lyrics first and labels each line — verse,
chorus, bridge — and flags when a chorus is a repeat of an earlier one,
so the orchestrator can reuse the same generated audio for that repeat
instead of regenerating it slightly differently each time.

**Why we need it:**
Full-song coherence — a listener's sense that a song "hangs together" —
depends on structure being planned deliberately, not emerging by accident
from generating everything as one undifferentiated stream of lyrics.

**Example:**
```
Input lyrics:
  [Verse 1]  Line A, Line B
  [Chorus]   Line C, Line D
  [Verse 2]  Line E, Line F
  [Chorus]   Line C, Line D   ← detected as a repeat of the first chorus

Structure planner output: Verse, Verse, Chorus, Verse, Verse, Chorus(repeat)
Composer: generates sung audio once for the Chorus, reuses it for the repeat
          rather than generating it twice (cheaper, and more consistent-sounding)
```

**Diagram:**
```
  full lyrics
        │
        ▼
  ┌───────────────┐
  │ Structure      │  labels each line: Verse / Chorus / Bridge
  │ planner        │  + detects repeated choruses
  └───────────────┘
        │
        ▼
  ┌───────────────┐
  │ Composer        │  dispatches each section to Singing Engine
  │ orchestration   │  (reusing audio for detected repeats),
  └───────────────┘  dispatches style prompt to Music Engine,
        │            informed by structure (bigger arrangement on chorus)
        ▼
  Mix (Phase 15) combines everything into the finished song
```

**Common beginner questions:**
- *Q: How does the planner know what a "chorus" is without being told?*
  → It's trained on real lyric sheets that already have section markers
  (`[Verse]`, `[Chorus]`) as standard formatting — a very common,
  freely-available source of supervision, so the model learns the
  pattern from thousands of real examples rather than hand-written rules.
- *Q: What if the user already knows the structure and wants to specify
  it directly?* → The full control layer (Phase 18) allows explicit
  section hints to override the planner's automatic detection.

---

## Phase 18: Full Control Layer

**Definition:** One single, fully-typed request format (`NaadRequest`)
that exposes every knob in the whole system — voice, emotion, melody,
music style, mix levels, output format, consent — with nothing hidden
behind an opaque preset.

**Beginner explanation:**
By this point in the roadmap, a dozen different subsystems each have
their own settings. This phase's job is to gather all of them into one
well-organized "form" a caller fills out, so using the system doesn't
require knowing which of twenty different function calls to make in
which order — one request, validated, dispatched correctly based on
which optional fields are filled in.

**Why we need it:**
Without a unified request type, every new feature added over time would
need its own bespoke API, and combining features (cloned voice + specific
emotion + specific melody + specific output format, all at once) would
become an ad-hoc mess rather than "just fill out more fields on the same
request."

**Example:**
```
NaadRequest {
    text: "some lyrics",
    voice: VoiceSpec::Cloned { reference: "ref.wav", consent: Some(token) },
    emotion: EmotionSpec { label: "joyful", intensity: 0.7 },
    singing: Some(SingingSpec { melody: melody_data }),
    background_music: Some(MusicSpec { style: "acoustic folk" }),
    output_format: AudioOutputFormat::Wav,
}

  → one request, dispatched automatically to Singing Engine (since
    `singing` is set) + Music Engine (since `background_music` is set)
    + Mix, all wired together correctly without the caller needing to
    know the internal call order
```

**Diagram:**
```
  one NaadRequest, every field explicit
          │
          ▼
  ┌───────────────┐
  │  Validation    │  (e.g. SingingSpec without lyrics → clear error)
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  Dispatch       │  routes to whichever engines the populated
  │                 │  fields require — Voice-only, full song, etc.
  └───────────────┘
          │
          ▼
  correct combination of engine calls, in the correct order
```

**Common beginner questions:**
- *Q: Why not just have a separate function for each combination of
  features?* → That approach doesn't scale — with a dozen optional
  features, the number of "combination" functions would explode. One
  request type with optional fields scales linearly instead.
- *Q: What happens with an invalid combination, like singing without
  lyrics?* → The validation step (Phase 18) catches it and returns a
  clear error — never a silent no-op and never a panic.

---

## Phase 19: Safety & Watermarking

**Definition:** The mechanisms that make sure voice cloning only happens
with the speaker's consent, and that every piece of generated audio can
be identified as AI-generated after the fact.

**Beginner explanation:**
Two separate protections. **Consent gating** means any request that
clones a real voice must carry a specific "yes, this person agreed"
token, or the system refuses — by default, not as an opt-in extra.
**Watermarking** embeds an inaudible signal into every piece of generated
audio (a pattern of sound outside the range a human notices, but
detectable by software) so that even if the audio is shared without any
label, a detector can later confirm "this was generated by this system."

**Why we need it:**
Voice cloning and singing-voice cloning are exactly the kind of
capability that can be misused for impersonation or fraud if built
without safeguards — this phase is what makes the difference between "a
tool that can clone voices" and "a tool that can clone voices
responsibly."

**Example:**
```
Request: clone reference speaker's voice, no consent token attached
  → REJECTED, clear error, no audio generated

Request: clone reference speaker's voice, valid consent token attached
  → generated audio produced, with an inaudible watermark embedded
  → running the same audio through the watermark detector later
    returns "watermarked: true", even after it's been mixed (Phase 15)
```

**Diagram:**
```
  cloning request
        │
        ▼
  ┌───────────────┐
  │ Consent check  │──── missing/invalid ────► REJECTED
  └───────────────┘
        │ valid
        ▼
  ┌───────────────┐
  │ Generate audio │
  └───────────────┘
        │
        ▼
  ┌───────────────┐
  │ Embed          │  inaudible spread-spectrum watermark
  │ watermark      │  (survives mixing/mastering)
  └───────────────┘
        │
        ▼
  final audio, cloneable-traceable even without an explicit label
```

**Common beginner questions:**
- *Q: Can the watermark be heard?* → No — it's specifically designed to
  sit in a part of the audio spectrum and at an amplitude level humans
  don't consciously notice, verified with a perceptual-difference test,
  not just assumed to be inaudible.
- *Q: Does watermarking survive editing, like mixing with music?* → Yes,
  that's a specific test in this phase's milestone — the watermark is
  checked for recoverability *after* passing through Phase 15's mixing
  pipeline, not just on raw, untouched output.

---

## Phase 20: Quantisation Stack

**Definition:** Shrinking a trained model's numbers down to lower
precision (e.g., from 32-bit to 4-bit) so it takes less memory and runs
faster, with only a small, measured quality cost.

**Beginner explanation:**
A model's weights are just enormous lists of numbers. Most of those
numbers don't need 32 bits of precision to do their job — a 4-bit
approximation is often nearly as good, and takes roughly 1/8th the
memory. This phase does that shrinking as a separate, deliberate step
after training, verified against the evaluation harness (Phase 24) to
make sure the quality cost is genuinely small before trusting the
quantised model for real use.

**Why we need it:**
On an 8GB laptop with no dedicated GPU, running a Medium-scale model in
full precision may simply not fit in memory alongside everything else
running — quantisation is what makes larger models usable on ordinary
hardware.

**Example:**
```
Medium-scale model, F32 weights: ~1.36 GB memory for inference
Same model, quantised to INT4:   ~178 MB memory for inference
                                  (roughly 1/8th the size)

Quality check: WER and speaker-similarity scores on a held-out
test set stay within a small, documented tolerance of the F32 version
```

**Diagram:**
```
  trained F32 checkpoint
          │
          ▼
  ┌───────────────┐
  │  Quantise       │  round each weight to a lower-precision value
  │  (INT8/INT4)    │
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  Evaluate       │  compare quality metrics vs. F32 baseline
  └───────────────┘
          │
          ▼
  smaller, faster checkpoint, quality cost measured and acceptable
```

**Common beginner questions:**
- *Q: Doesn't shrinking the numbers make the model noticeably worse?* →
  Some quality is lost, but usually much less than you'd expect — most
  of a model's numbers carry more precision than they actually need, and
  this phase's whole job is to measure exactly how much is lost and
  confirm it's within an acceptable range before shipping a quantised
  checkpoint.
- *Q: Why not just always use INT4 then?* → There's a real tradeoff — the
  smaller the numbers, the faster and lighter the model, but the bigger
  the risk of quality loss. INT8 is a safer middle ground; INT4 is used
  where memory constraints (like the i3 laptop) make the tradeoff worth
  it.

---

## Phase 21: Fine-Tuning Refinement (LoRA / QLoRA / DoRA)

**Definition:** Deliberately adapting a trained model to a specific
speaker, singing style, or accent using a real, curated dataset — the "do
it properly, with real data" counterpart to Phase 23's lightweight online
learning.

**Beginner explanation:**
Rather than retraining the entire model (expensive, slow, needs huge
amounts of data), LoRA-family methods add a small number of new,
trainable parameters alongside the mostly-frozen original model — the
same trick `aarambh-studio` uses for text fine-tuning, applied here to
voice/style adaptation. QLoRA does the same thing on a quantised (Phase
20) base model to save even more memory; DoRA is a refinement of LoRA
that tends to match full fine-tuning quality more closely.

**Why we need it:**
Zero-shot cloning (Phase 8) works from a single short clip, but for a
speaker you'll use extensively and want to sound as good as possible, a
proper fine-tune with more data (this phase) will generally outperform
it — depth of adaptation the zero-shot path can't reach from 8 seconds
of audio alone.

**Example:**
```
aarambh-voice-studio finetune --recipe speaker_adapt \
  --data speaker_clips/ --out adapter.safetensors

  → produces a small adapter file (much smaller than the full model)
    that, when loaded alongside the base model, measurably improves
    speaker-similarity scores for this specific speaker compared to
    zero-shot cloning from a single reference clip
```

**Diagram:**
```
  base model (mostly frozen)
          │
          ▼ + small LoRA adapter (newly trained, few parameters)
  ┌───────────────┐
  │  Combined       │  behaves like a specialized model for this speaker
  │  model          │  without retraining the whole thing
  └───────────────┘
```

**Common beginner questions:**
- *Q: How is this different from Phase 23's self-learning?* → This phase
  is deliberate, offline, and uses a properly curated dataset — you
  decide when to run it. Phase 23 happens automatically at the moment a
  new sample is provided, using a much smaller and cheaper update. Both
  exist because they solve different problems.
- *Q: Why three different methods (LoRA, QLoRA, DoRA) instead of just
  one?* → They trade off memory usage against how closely they match full
  fine-tuning quality — QLoRA is the most memory-efficient (good for
  Kaggle's free GPU limits), DoRA tends to be the highest quality, LoRA
  is the well-tested middle ground.

---

## Phase 22: Alignment — GRPO + DPO

**Definition:** A training step, after ordinary fine-tuning, that pushes
the model toward generating audio that scores well on the metrics you
actually care about — not just "predicts the next token correctly," but
"sounds natural, keeps the right speaker identity, matches the requested
emotion."

**Beginner explanation:**
Up through Phase 21, the model is trained to minimize prediction error
against real recordings — a proxy for quality, but not quality itself.
This phase adds a second training stage that directly optimizes for
quality, using the same metrics the evaluation harness (Phase 24) already
computes as a *reward signal*. GRPO does this by generating several
candidate outputs for the same prompt, scoring each, and nudging the
model toward whichever candidates scored best relative to the group. DPO
does something similar more cheaply, by building (better, worse) pairs
from sampled candidates and training the model to prefer the better one,
without needing a live generation loop during training.

**Why we need it:**
A model that's only ever been trained to match training recordings
exactly can still leave real quality on the table — GRPO/DPO close that
gap by optimizing for the thing you actually want (naturalness, fidelity)
rather than a proxy for it.

**Example:**
```
Prompt: "say this sentence sadly"

Generate 6 candidate clips → score each with the emotion-accuracy metric
Candidates 1, 2 score highest → "chosen" for DPO
Candidates 5, 6 score lowest  → "rejected" for DPO

DPO training nudges the model toward whatever made candidates 1, 2
sound more convincingly sad than candidates 5, 6
```

**Diagram:**
```
  same prompt, sampled K times
          │
          ▼
  ┌───────────────┐
  │  Score each     │  using aarambh-voice-eval metrics
  │  candidate      │  (WER, speaker-sim, emotion-acc, MOS-proxy...)
  └───────────────┘
          │
   GRPO: reinforce above-group-average, discourage below-average
   DPO:  pick (best, worst) pair, train to prefer the best
          │
          ▼
  model shifted toward higher-scoring generations
```

**Common beginner questions:**
- *Q: Why not just train longer with the original loss instead?* →
  Because the original loss only measures "did you predict the exact
  tokens from the training recording," which caps out at reproducing the
  training data's *average* quality — GRPO/DPO can push past that by
  directly rewarding what's actually being measured as good.
- *Q: Could this alignment step make the model worse at something else,
  like intelligibility, while it's busy improving naturalness?* → It's a
  real risk, which is why this phase's tests explicitly check that WER
  and speaker-similarity don't regress — alignment is only accepted if it
  improves the target metric without quietly breaking another one.

---

## Phase 23: Self-Learning

**Definition:** Letting the deployed system learn a new voice or fix a
mistake from a single example, safely, without a full retraining job, and
without forgetting voices it already knows.

**Beginner explanation:**
Fine-tuning (Phase 21) is deliberate and offline — you decide when to run
it, on what data. Self-learning is the opposite: it happens automatically,
at the moment someone provides a new sample, using a much smaller and
cheaper update. Two ideas make this safe. First, **gradient
orthogonalization** — a mathematical trick that makes the update to a
*new* voice provably not interfere with voices already learned, the same
anti-forgetting mechanism used in the Manas project. Second, a
**confidence gate** — the update is only kept if it's checked against the
evaluation harness and doesn't make anything worse; if it does, it's
silently discarded and the live model is untouched.

**Why we need it:**
Without this, every new voice or every small correction ("that clone
sounded slightly off") would require a full offline fine-tuning job —
impractical for something as small as "remember this one new voice from 8
seconds of audio."

**Example:**
```
aarambh-voice-studio learn --sample new_voice.wav --identity-hint "warm, mid-30s"

  → computes a small adapter update for this new voice
  → checks: does this update, staged but not yet committed, score at
    least as well on the evaluation harness as before?
  → YES: committed — the system now "knows" this voice for future requests
  → NO:  discarded — live model completely unchanged, reason logged
```

**Diagram:**
```
  new sample arrives
          │
          ▼
  ┌───────────────┐
  │ Lookup: known   │  is this speaker/style already known?
  │ or new voice?   │
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │ Orthogonalize   │  make sure this update can't disturb
  │ the update      │  voices already learned
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │ Confidence      │──── fails ────► discard, live model untouched
  │ gate            │
  └───────────────┘
          │ passes
          ▼
  committed — new voice remembered, existing voices unaffected
```

**Common beginner questions:**
- *Q: How can you be sure learning voice #50 doesn't quietly ruin voice
  #1?* → This is specifically tested — a regression test runs 50
  sequential updates for 50 distinct speakers, then re-checks speaker #1's
  quality after all 50 updates, confirming gradient orthogonalization
  actually held up in practice, not just in theory.
- *Q: What if someone submits a bad or noisy sample?* → The confidence
  gate is the safety net — a bad sample that would make things worse
  simply fails the gate and gets discarded, no special-casing needed for
  "detecting bad input" ahead of time.

---

## Phase 24: Evaluation Harness + Baseline Comparison

**Definition:** A standardized set of automatic tests that measure how
good the generated audio actually is — intelligibility, speaker fidelity,
emotion accuracy, musical tag accuracy, overall naturalness — plus a
comparison against other publicly available systems, so "good" means
something specific.

**Beginner explanation:**
Without a consistent measurement, "does this sound better now?" is just a
feeling. This phase builds actual numbers: transcribing generated speech
back to text and comparing to what was asked for (catches unintelligible
output), comparing a cloned voice's "fingerprint" to the reference
(catches bad cloning), and so on — plus running the same test prompts
against a few well-known open-source systems, so you have an external
reference point, not just "better than my last checkpoint."

**Why we need it:**
Every later decision — is this fine-tune worth keeping, did this
alignment run actually help, is this quantised model still good enough —
depends on having a trustworthy, repeatable measurement to check against.

**Example:**
```
aarambh-voice-studio eval --all --with-baseline

Scorecard:
  TTS intelligibility (WER):        3.2%   (baseline system: 4.1%)
  Cloning speaker-similarity:       0.87   (baseline system: 0.81)
  Emotion accuracy:                 78%    (baseline system: n/a)
  Music tag agreement:              0.71   (baseline system: 0.68)
  MOS-proxy (naturalness):          3.9/5  (baseline system: 4.1/5)
```

**Diagram:**
```
  test prompts (fixed set, held out from training)
          │
          ▼
  ┌───────────────┐
  │  Generate with  │  your own checkpoint
  │  this project   │
  └───────────────┘         ┌───────────────┐
          │                  │  Generate with  │  external baseline
          ▼                  │  reference       │  system(s)
  ┌───────────────┐         └───────────────┘
  │  Score with     │                  │
  │  eval metrics   │◄─────────────────┘
  └───────────────┘
          │
          ▼
  scorecard: your numbers, baseline numbers, side by side
```

**Common beginner questions:**
- *Q: Why compare against other systems at all, instead of just your own
  progress over time?* → "Better than my last checkpoint" can still mean
  "still far behind what's actually possible" — an external baseline
  tells you whether you're closing a real gap or just making small
  improvements in a vacuum.
- *Q: Is the MOS-proxy as good as asking real humans to rate the audio?*
  → It's a useful, fast approximation, trained to predict what human
  raters would likely say — not a perfect substitute, but far more
  practical than collecting human ratings for every single checkpoint.

---

## Phase 25: GPU Scale-Up + Speculative Decoding

**Definition:** Training larger (Small/Medium/Large) versions of every
model on Kaggle's free GPUs, and fully activating the speculative
decoding scaffold from Phase 7 to speed up generation.

**Beginner explanation:**
Everything up to this point has been provable at Tiny scale on an
ordinary laptop. This phase is where the same recipes get run at larger
scale on borrowed GPU time. Speculative decoding gets finished here too: a
small "draft" checkpoint quickly guesses several tokens ahead, and the
real, bigger model checks all of those guesses in a single pass rather
than one token at a time — when the guesses are right (which is often),
generation finishes noticeably faster with no change in output quality.

**Why we need it:**
Tiny-scale models prove the mechanisms work, but real usable audio
quality generally needs more parameters than an i3 laptop can train from
scratch in reasonable time — this phase is where quality actually gets to
production-usable levels.

**Example:**
```
Without speculative decoding: generate 100 tokens = 100 forward passes
                               through the full Medium-scale model

With speculative decoding: a Tiny "draft" model quickly guesses 4 tokens
                            ahead; the Medium model verifies all 4 in
                            one pass; if all 4 are accepted, that's
                            75% fewer full-model forward passes needed
```

**Diagram:**
```
  Tiny draft model  ──►  guesses tokens [A, B, C, D]
                                  │
                                  ▼
  Medium target model  ──►  verifies all 4 in one forward pass
                                  │
                    accepted prefix (e.g. A, B, C) kept,
                    generation continues from there
```

**Common beginner questions:**
- *Q: Does speculative decoding ever produce worse output, since a
  smaller model is "guessing"?* → No — the target model always verifies
  the draft's guesses; anything not accepted is regenerated correctly by
  the target model itself. Output is provably identical to not using
  speculative decoding at all — it's purely a speed optimization.
- *Q: Why train on Kaggle instead of buying more compute?* → Part of this
  project's philosophy (matching `aarambh-studio`) is proving the whole
  pipeline works using free, ordinary hardware — an i3 laptop plus
  Kaggle's free GPU tier — rather than assuming access to paid compute.

---

## Phase 26: Inference Server + Audio Output Formats

**Definition:** A network server that accepts requests over HTTP and
streams generated audio back, in whichever file format the caller wants.

**Beginner explanation:**
Up to now, everything has run as a one-off command-line call. A server
makes the system usable by other programs (a website, an app) over a
network connection, and can serve multiple people's requests
concurrently. This phase also finalizes exactly which audio formats come
out the other end — WAV for lossless compatibility, FLAC for smaller
lossless files, Opus for efficient streaming, and MP3 for legacy
compatibility (kept optional due to its licensing terms).

**Why we need it:**
A command-line tool that only one person can run one request at a time on
their own laptop isn't how most people will actually want to use this
system — a server is what makes it usable as a real product.

**Example:**
```
POST /generate
{
  "text": "hello from the server",
  "voice": {"preset": "neutral"},
  "output_format": "opus"
}

  → server streams back Opus-encoded audio, chunk by chunk, as it's
    generated, rather than waiting for the whole clip to finish first
```

**Diagram:**
```
  HTTP request (NaadRequest as JSON)
          │
          ▼
  ┌───────────────┐
  │  Server          │  batches concurrent requests together
  │  (axum)          │
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  Generate +      │  streams audio frame-by-frame as it's produced
  │  stream          │
  └───────────────┘
          │
          ▼
  response encoded in the requested format (WAV/FLAC/Opus/MP3)
```

**Common beginner questions:**
- *Q: Why is Opus the default for streaming instead of WAV?* → Opus
  produces much smaller output for a given perceived quality, which
  matters a lot when streaming audio over a network connection — WAV
  remains available for anyone who explicitly wants lossless output.
- *Q: Why is MP3 support "optional" instead of just included?* → MP3
  encoding (via the LAME library) carries its own licensing terms
  separate from this project's Apache-2.0 license — keeping it behind a
  feature flag means anyone building the project makes an informed choice
  about including it, rather than it being silently bundled by default.

---

## Phase 27: Production Release v1.0

**Definition:** The final quality pass — documentation, continuous
integration checks, a full evaluation scorecard — and the tag that says
"this is the finished v1."

**Beginner explanation:**
This phase doesn't add new capability. It makes sure everything built
across the previous 27 phases is documented, tested automatically on
every change, and measured one final time against the evaluation
harness, so the v1.0.0 release is something a stranger could clone,
build, and understand without having watched the whole roadmap unfold.

**Why we need it:**
A pile of working code isn't the same thing as a releasable project —
this phase is the difference between "it works on my machine" and "here
is a tagged, documented, tested v1.0.0."

**Example:**
```
git checkout v1.0.0
cargo build --release -p aarambh-voice-studio
target/release/aarambh-voice-studio --version
  → aarambh-voice-studio 1.0.0

Every CLI subcommand documented in the README runs successfully
against a Tiny checkpoint on a clean environment.
```

**Diagram:**
```
  all 27 previous phases' code
          │
          ▼
  ┌───────────────┐
  │ Docs pass        │  every pub item documented, per-crate READMEs
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │ CI checks        │  fmt, clippy, tests, docs — all must pass
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │ Final scorecard  │  one last evaluation-harness run, published
  └───────────────┘
          │
          ▼
  tag v1.0.0
```

**Common beginner questions:**
- *Q: Why run the evaluation harness one more time here, if it's already
  been run throughout the roadmap?* → This final run is the number that
  actually ships with the release — it's the answer to "how good is
  v1.0.0, specifically," rather than a snapshot from partway through
  development.
- *Q: Does v1.0.0 include pretrained checkpoints?* → No — per the
  project's source-first release policy, v1.0.0 is a source and
  engineering release; checkpoints, voice packs, and adapters are never
  bundled by default.

---

*That's all 28 phases. For the exact math behind the loss functions,
embeddings, and training algorithms mentioned throughout both parts, see
`VOICE_STUDIO_MATH_FORMULAS_GUIDE_PART1.md` / `PART2.md`. For where the
training data for each phase actually comes from, and the audio-specific
terminology used throughout, see
`VOICE_STUDIO_AUDIO_ML_TERMINOLOGY_AND_DATASET_GUIDE.md`.*
