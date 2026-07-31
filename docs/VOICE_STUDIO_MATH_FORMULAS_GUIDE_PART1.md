# aarambh-voice-studio: The Complete Math Formula Guide (Part 1 of 2)

### Every formula we use, explained like you've never seen math notation before

This document is for someone coming from a **non-math background** —
maybe you know how to code, but formulas with Greek letters and weird
symbols look scary. That's fine. Every formula below is broken down
piece-by-piece, like reading a sentence word-by-word, before we ever
touch a real number.

For every formula, you'll get:
- **What it's called** (definition)
- **How to read the symbols** (a beginner "translation" of the notation)
- **Why we use it in aarambh-voice-studio** (which phase it belongs to)
- **The formula itself**
- **2 fully solved examples**, step by step, with real numbers
- **A beginner question**

This file covers Phases 1–13's math. Part 2 covers the math behind
Phases 13–23 (singing, mixing, alignment, self-learning).

---

## How to Read Any Formula (read this first!)

Here's a "decoder ring" for symbols that show up again and again. Keep
coming back to this section whenever a symbol confuses you.

| Symbol | Say it as | Meaning |
|---|---|---|
| `Σ` (sigma) | "sum of" | Add up a bunch of things |
| `x` | "x" | Usually the input, or a raw waveform sample |
| `y` | "y" | Usually the true/target value |
| `ŷ` (y-hat) | "y hat" | The model's *predicted* value |
| `e` (in codec context) | "e" | A codebook entry (a "dictionary word" for sound) |
| `θ` (theta) | "theta" | A general stand-in for "all the model's learnable numbers" |
| `β` (beta) | "beta" | A weighting constant, tuned by hand |
| `α` (alpha) | "alpha" | A small scaling constant |
| `E[·]` | "expected value of" | "The average, over a batch of examples" |
| `exp(x)` or `eˣ` | "e to the x" | A way of making numbers grow fast (~2.718 raised to power x) |
| `log(x)` | "log of x" | The opposite of exp — "undoes" exponential growth, measures surprise/error |
| `‖x‖` | "norm of x" | Roughly: "how big is this whole vector, as one single size number" |
| `‖x‖²` | "squared norm of x" | The norm, squared — sum of each number squared |
| `‖x‖₁` | "L1 norm of x" | Sum of *absolute values* — more forgiving of big single errors than squaring |
| `σ(x)` | "sigmoid of x" | Squashes any number into the range between 0 and 1 |
| `cos(a, b)` | "cosine similarity" | How aligned two vectors are in direction, from -1 to 1 |
| `sg[x]` | "stop gradient of x" | "Treat this as a fixed constant, don't backpropagate through it" |

You don't need to memorize this table — just refer back to it whenever a
symbol below looks unfamiliar.

---

## 1. Waveform & Sample Rate (the raw material — no formula yet, just the setup)

**Definition:** A waveform is simply a long list of numbers, one per
tiny moment in time, each representing air pressure (loudness) at that
instant.

**How to read it:** A "24kHz waveform" means 24,000 of these numbers
represent one second of sound.

**Why we use it:** Every single phase in this project — the codec,
attention, training losses — ultimately operates on this list of numbers
or something derived from it. It's the audio equivalent of raw text
before tokenization in `aarambh-studio`.

**Example 1 (a very short waveform):**
```
0.1 seconds of audio at 24kHz = 2,400 numbers

first few values: [0.00, 0.02, 0.05, 0.03, -0.01, -0.04, ...]
```

**Example 2 (comparing sample rates):**
```
1 second at 16kHz (common for some speech tasks): 16,000 numbers
1 second at 24kHz (this project's speech codec):  24,000 numbers
1 second at 44.1kHz (music, "CD quality"):        44,100 numbers

Same 1 second of real time, very different amounts of raw data —
this is exactly why the codec (formula 2 below) exists: to shrink
this down to something a transformer can actually work with.
```

**Beginner question:** *Why not just always use the highest sample rate
for everything?* Higher sample rates capture more detail (useful for
music, which has more high-frequency content humans care about), but
also mean more numbers to process for every second of audio — a direct
tradeoff between fidelity and compute cost, which is why this project
uses 24kHz for speech/singing and 44.1kHz for music rather than one rate
for everything.

---

## 2. Short-Time Fourier Transform (STFT) & Mel Spectrogram

**Definition:** STFT takes a waveform and produces a picture of it: time
along one axis, frequency along the other, with brightness showing how
strong each frequency is at each moment.

**How to read it:**
```
STFT(x)[t, f] = Σ_n  x[n] · w[n - t·H] · e^(-i·2π·f·n/N)
```
Say it as: "the strength of frequency f, at time-chunk t, is found by
taking a small window of the waveform, multiplying it by a smooth
fade-in/fade-out shape, and measuring how much of frequency f is present
in that windowed chunk."

**Why we use it:** The codec's discriminators (formula 5) and the music
understanding classifier (Phase 11) don't look at raw waveforms directly
— they look at this time-vs-frequency picture, because patterns like
"this is a snare drum hit" or "this is the vowel sound in 'cat'" are much
easier to recognize as shapes in a spectrogram than as a wiggly line of
raw numbers.

**Formula:**
```
STFT(x)[t, f] = Σ_n  x[n] · w[n - t·H] · e^(-i·2π·f·n/N)
```
- `x[n]` — the waveform, sample number n.
- `w[n]` — a window function (fades in/out smoothly, avoiding harsh
  cutoff artifacts).
- `H` — hop length: how far the window slides forward each step.
- `N` — FFT size: how many samples analyzed at once.
- `t` — which time-chunk (frame).
- `f` — which frequency we're measuring.

**Example 1 (counting frames):**
```
1-second clip, 24kHz (24,000 samples), N = 1024, H = 256

number of frames ≈ 24000 / 256 ≈ 93 frames
each frame has N/2 + 1 = 513 frequency values (only half is unique
  for real-valued audio)

Result: a 93 × 513 grid — this grid, not the raw waveform, is what
the codec's discriminator and the music classifier actually see.
```

**Example 2 (what changing the window size does):**
```
Small N (e.g. 256): fewer frequency bins, but very precise about *when*
  something happened — good for catching sharp transients like a
  drum hit
Large N (e.g. 2048): many more frequency bins, very precise about
  *which pitch*, but blurs together things that happened close in time

This project's multi-resolution STFT loss (formula 4) deliberately
uses several different N values at once, so it catches both kinds
of detail rather than picking just one.
```

**Beginner question:** *Do I need to compute this by hand?* Never — the
`rustfft` crate computes the actual Fourier transform. What matters for
understanding this project is the *output*: a grid of numbers describing
"how much of each frequency, at each moment," which every audio
classifier and discriminator in this codebase is built to look at.

---

## 3. Mel Scale (re-binning the spectrogram)

**Definition:** The mel scale takes a regular (linear) frequency
spectrogram and re-bins it to match how human hearing actually perceives
pitch differences.

**How to read it:**
```
mel(f) = 2595 · log10(1 + f/700)
```
Say it as: "take the real frequency f in Hz, and convert it to a 'mel'
value using this formula — the result compresses high frequencies much
more than low ones."

**Why we use it:** Humans are much better at telling apart 100Hz from
200Hz than telling apart 10,000Hz from 10,100Hz, even though both pairs
are "100Hz apart." Training the music classifier and codec loss on a
mel-scaled spectrogram instead of a raw linear one focuses the model's
attention on the frequency differences that actually matter perceptually.

**Formula:**
```
mel(f) = 2595 · log10(1 + f/700)
```

**Example 1 (a low frequency):**
```
f = 100 Hz
mel(100) = 2595 · log10(1 + 100/700)
         = 2595 · log10(1.143)
         = 2595 · 0.0580
         ≈ 150.5 mel
```

**Example 2 (a much higher frequency):**
```
f = 10,000 Hz
mel(10000) = 2595 · log10(1 + 10000/700)
           = 2595 · log10(15.29)
           = 2595 · 1.184
           ≈ 3072 mel

Notice: going from 100Hz→10,000Hz (a 9,900 Hz jump) only moved
about 2,921 mel units, while a much smaller jump at low frequencies
would move proportionally more — the compression at high frequencies
that matches human hearing.
```

**Beginner question:** *Does this mean high notes just get ignored?* No
— they're still represented, just compressed relative to low
frequencies, matching the fact that a listener's *perceptual* sensitivity
to a given Hz-difference shrinks as frequency rises. Nothing is thrown
away; the scale is just non-linear.

---

## 4. Vector Quantization (VQ) Commitment Loss

**Definition:** A loss that pulls the codec encoder's raw output toward
the nearest "dictionary" entry (codebook), and pulls that codebook entry
toward the encoder's output, so the two meet in the middle.

**How to read it:**
```
L_vq = || sg[z_e] - e ||²  +  β · || z_e - sg[e] ||²
```
Say it as: "the squared distance between the encoder's output and its
nearest codebook entry, counted twice — once training the codebook to
move toward the encoder, once (weighted by β) training the encoder to
move toward the codebook."

**Why we use it:** This is the core mechanism of Phase 1's codec — it's
what turns a continuous, infinite-possibilities encoder output into a
fixed vocabulary of discrete tokens the transformer can predict, the same
way a text tokenizer turns infinite possible words into a fixed
vocabulary.

**Formula:**
```
L_vq = || sg[z_e] - e ||²  +  β · || z_e - sg[e] ||²
```

**Example 1 (encoder output close to its codebook entry):**
```
z_e = [1.0, 2.0],  nearest codebook entry e = [0.8, 2.2],  β = 0.25

|| z_e - e ||² = (1.0-0.8)² + (2.0-2.2)² = 0.04 + 0.04 = 0.08

L_vq = 0.08 (codebook term) + 0.25 × 0.08 (encoder term)
     = 0.08 + 0.02 = 0.10   ← small loss, they were already close
```

**Example 2 (a bad match, far apart):**
```
z_e = [1.0, 2.0],  nearest codebook entry e = [3.0, -1.0],  β = 0.25

|| z_e - e ||² = (1.0-3.0)² + (2.0-(-1.0))² = 4.0 + 9.0 = 13.0

L_vq = 13.0 + 0.25 × 13.0 = 13.0 + 3.25 = 16.25   ← much bigger loss,
       pushing both encoder output and codebook entry to move
       toward each other more aggressively next training step
```

**Beginner question:** *Why does β only apply to one of the two terms?*
The two terms have different jobs: the first term trains the codebook
(so it doesn't move too fast, β=1 implicitly), while the second term
trains the encoder (β is usually kept smaller, like 0.25, so the encoder
doesn't chase a constantly-shifting codebook too eagerly — a classic
stability trick from the original VQ-VAE paper this idea comes from).

---

## 5. Reconstruction Loss (L1 + Multi-Resolution STFT)

**Definition:** A loss comparing the codec's reconstructed waveform
against the real one, both directly (sample by sample) and in spectrogram
form at several different resolutions.

**How to read it:**
```
L_recon = || x - x̂ ||₁  +  Σ_r  [ || STFT_r(x) - STFT_r(x̂) ||₁
                                 + || log STFT_r(x) - log STFT_r(x̂) ||₁ ]
```
Say it as: "add up the absolute difference between the real and
reconstructed waveform, plus — for several different STFT window sizes r
— the absolute difference between their spectrograms, plus the absolute
difference between their *log*-scaled spectrograms."

**Why we use it:** This is the most direct "does the reconstruction sound
like the original" signal in the whole codec — everything else (VQ loss,
adversarial loss) supports this core goal, but this loss is the one
comparing actual sound content most literally.

**Formula:**
```
L_recon = || x - x̂ ||₁  +  Σ_r  [ || STFT_r(x) - STFT_r(x̂) ||₁
                                 + || log STFT_r(x) - log STFT_r(x̂) ||₁ ]
```

**Example 1 (waveform term, a few samples):**
```
real x =        [0.10, 0.20, -0.05]
reconstructed x̂ = [0.12, 0.18, -0.04]

|| x - x̂ ||₁ = |0.10-0.12| + |0.20-0.18| + |-0.05-(-0.04)|
             = 0.02 + 0.02 + 0.01 = 0.05
```

**Example 2 (why the log term matters, quiet passage):**
```
A quiet passage: real spectrogram value = 0.002, reconstructed = 0.006

Linear difference: |0.002 - 0.006| = 0.004   ← looks tiny, easy to ignore

Log difference: |log(0.002) - log(0.006)| = |-6.21 - (-5.12)| = 1.09
  ← in log space, this is actually a huge relative error (the
    reconstruction is 3x louder than it should be in a quiet section)

This is exactly why the log-STFT term exists: it catches quiet-passage
errors the plain linear term would barely notice.
```

**Beginner question:** *Why use several different resolutions (the Σ_r
sum) instead of just one STFT setting?* A single window size is a
tradeoff between time-precision and frequency-precision (see formula 2,
example 2) — using several resolutions at once means the loss catches
both sharp transient errors (drum hits, consonants) and slower harmonic
errors (sustained notes, vowel quality) that a single resolution would
miss.

---

## 6. Adversarial (GAN) Hinge Loss & Feature Matching

**Definition:** A loss from a second, competing model (the discriminator)
whose only job is telling real audio apart from the codec's
reconstructions — the codec is trained to fool it.

**How to read it (discriminator):**
```
L_D = E[ max(0, 1 - D(x)) ]  +  E[ max(0, 1 + D(x̂)) ]
```
Say it as: "the discriminator is penalized if it's not confident enough
that real audio is real, and penalized if it's not confident enough that
reconstructed audio is fake."

**How to read it (generator/codec side):**
```
L_G_adv = -E[ D(x̂) ]
L_feat  = Σ_l || D_l(x) - D_l(x̂) ||₁
```
Say it as: "the codec is rewarded for making the discriminator's
verdict on the reconstruction as 'real-looking' (high) as possible, and
additionally rewarded for making the discriminator's *internal* reactions
to real and reconstructed audio match at every layer l."

**Why we use it:** Adversarial training pushes the codec toward
reconstructions that are perceptually convincing, not just numerically
close — the same idea used in GANs for images, applied here to audio.
Feature matching adds a gentler, more stable training signal on top,
which tends to make this kind of adversarial training much less flaky in
practice.

**Formula:**
```
L_D = E[ max(0, 1 - D(x)) ] + E[ max(0, 1 + D(x̂)) ]
L_G_adv = -E[ D(x̂) ]
L_feat  = Σ_l || D_l(x) - D_l(x̂) ||₁
```

**Example 1 (discriminator doing well):**
```
D(x) = 0.9   (correctly confident real audio is real)
D(x̂) = -0.3  (correctly suspects reconstruction is fake)

L_D = max(0, 1-0.9) + max(0, 1+(-0.3)) = 0.1 + 0.7 = 0.8
```

**Example 2 (discriminator being fooled — good for the codec, bad for the discriminator):**
```
D(x) = 0.9    (still correctly confident about real audio)
D(x̂) = 0.85   (discriminator now thinks the fake is ALSO probably real)

L_D = max(0, 1-0.9) + max(0, 1+0.85) = 0.1 + 1.85 = 1.95
      (a much bigger loss for the discriminator — it's being fooled)

L_G_adv = -0.85 = -0.85
      (a very favorable, low loss for the codec/generator — it
       successfully fooled the discriminator this time)
```

**Beginner question:** *Doesn't training two models against each other
risk one "winning" and the whole thing breaking down?* Yes, this is a
well-known GAN training risk — it's exactly why feature matching (`L_feat`)
is included as a gentler, more stable additional signal, and why the
codec's freeze criterion (Phase 1) requires checking several different
quality metrics rather than trusting the adversarial loss curve alone.

---

## 7. Semantic Distillation (Cosine Distance)

**Definition:** A loss that trains part of the codec's tokens (RVQ-1) to
match a separate, frozen "teacher" model's understanding of the audio's
linguistic content — not just its raw acoustic shape.

**How to read it:**
```
L_sem = 1 - ( (v · t) / (||v|| · ||t||) )
```
Say it as: "one minus the cosine similarity between the codec's semantic
embedding v and the frozen teacher model's feature t for the same audio
— so a perfect match gives zero loss."

**Why we use it:** This is what makes this project's codec more than just
a compression scheme — it's what makes RVQ-1's tokens carry meaning
(closer to "phoneme-ish" content) rather than only fine acoustic detail,
which in turn makes the downstream transformer's job of predicting the
next token genuinely easier.

**Formula:**
```
L_sem = 1 - ( (v · t) / (||v|| · ||t||) )
```

**Example 1 (well-aligned vectors):**
```
v = [1, 0],  t = [1, 1]

v · t = 1×1 + 0×1 = 1
||v|| = √(1²+0²) = 1
||t|| = √(1²+1²) ≈ 1.414

cosine similarity = 1 / (1 × 1.414) ≈ 0.707
L_sem = 1 - 0.707 = 0.293
```

**Example 2 (perfectly aligned vectors — best case):**
```
v = [1, 0],  t = [1, 0]   (identical direction)

v · t = 1×1 + 0×0 = 1
||v|| = 1,  ||t|| = 1
cosine similarity = 1 / (1×1) = 1.0

L_sem = 1 - 1.0 = 0.0   ← zero loss, the codec's semantic codes
                          already perfectly match the teacher's
                          understanding of this audio
```

**Beginner question:** *What exactly is the "teacher" model?* It's a
separate, already-trained self-supervised speech model (not trained
jointly with this project's codec) whose features are computed once and
cached ahead of time — the codec never trains the teacher, only learns to
match it, the same conceptual relationship as a student studying a
textbook that was written independently.

---

## 8. Scaled Dot-Product Attention (shared with aarambh-studio)

**Definition:** The mechanism that lets the model decide, for every
token, which other tokens in the sequence matter most for predicting what
comes next.

**How to read it:**
```
Attention(Q, K, V) = softmax( (Q·Kᵀ) / √d_k ) · V
```
Say it as: "compare every query against every key with a dot product,
scale it down, turn the results into probabilities with softmax, then use
those probabilities to blend together the values."

**Why we use it:** Identical mechanism to `aarambh-studio`'s text attention,
applied here to codec-token sequences instead of word tokens — it's what
lets the model know, when generating audio token #200, which of the
previous 199 tokens (and any conditioning like a speaker embedding) are
most relevant right now.

**Formula:**
```
Attention(Q, K, V) = softmax( (Q·Kᵀ) / √d_k ) · V
```

**Example 1 (simplified, 2 tokens, 1D):**
```
Q = [1.0], K = [1.0, 0.5], V = [10.0, 2.0],  d_k = 1

raw scores = Q·K = [1.0×1.0, 1.0×0.5] = [1.0, 0.5]
scaled = [1.0/√1, 0.5/√1] = [1.0, 0.5]
softmax([1.0, 0.5]) ≈ [0.622, 0.378]

output = 0.622×10.0 + 0.378×2.0 = 6.22 + 0.756 = 6.976
```

**Example 2 (what happens with a much bigger score gap):**
```
Q = [2.0], K = [1.0, 0.1], V = [10.0, 2.0],  d_k = 1

raw scores = [2.0×1.0, 2.0×0.1] = [2.0, 0.2]
softmax([2.0, 0.2]) ≈ [0.858, 0.142]

output = 0.858×10.0 + 0.142×2.0 = 8.58 + 0.284 = 8.864

Notice: the bigger the gap between scores, the more sharply attention
focuses on the higher-scoring token — this is softmax's job, turning
raw comparisons into a "mostly focus here, a little bit there" blend.
```

**Beginner question:** *Why divide by √d_k before softmax?* Without this
scaling, dot products of long vectors tend to get very large, which
pushes softmax toward extremely sharp, nearly all-or-nothing outputs —
dividing by √d_k keeps the scores in a more reasonable range so training
stays stable. Full derivation in `aarambh-studio-math-formulas-guide.md` §4.

---

## 9. RoPE & Grouped-Query Attention — summary

**RoPE (Rotary Position Embedding)** rotates each query/key vector by an
angle depending on its position in the sequence, so attention naturally
"knows" how far apart two tokens are without a separate learned position
embedding. Identical formula to `aarambh-studio`'s RoPE — see that guide's
full rotation-matrix derivation. Applied here to codec token positions
(time steps in the audio) instead of word positions.

**Grouped-Query Attention (GQA)** lets several query "heads" share one
Key/Value pair instead of each head having its own — cutting memory and
compute with a small, usually acceptable, quality tradeoff.
`ModelConfig`'s `n_kv_heads < n_heads` controls how much sharing happens.

**Example (GQA, concrete numbers):** A Small-scale model has `n_heads=8`
and `n_kv_heads=4` — meaning every 2 query heads share 1 Key/Value pair,
roughly halving the memory needed to store Keys and Values during
generation compared to `n_kv_heads=8` (no sharing at all).

**Beginner question:** *Doesn't sharing Key/Value pairs lose information?*
Some, in principle — but in practice the quality cost is usually small
relative to the memory and speed savings, which is why GQA is standard
in modern efficient transformer designs rather than a niche trick.

---

## 10. Cross-Entropy Loss (token prediction)

**Definition:** The loss that measures how much probability the model
assigned to the *correct* next codec token, penalizing low confidence in
the right answer.

**How to read it:**
```
L_CE = -Σ_i  y_i · log(ŷ_i)
```
Say it as: "for every possible token i, multiply whether it's the true
answer (1 or 0) by the log of the model's predicted probability for it,
then sum and negate." Since `y_i` is 0 everywhere except the correct
token, this simplifies to just `-log(ŷ_correct)`.

**Why we use it:** This is the core training signal for every
autoregressive phase in this project — TTS (Phase 6), music generation
(Phase 12), singing (Phase 13) — all trained by comparing predicted codec
tokens against real ones with this exact loss.

**Formula:**
```
L_CE = -log(ŷ_correct)
```

**Example 1 (low confidence in the right answer):**
```
Codebook has 2048 possible tokens.
Model assigns probability 0.02 to the correct one.

L_CE = -log(0.02) ≈ 3.91   ← a high loss, the model wasn't very sure
```

**Example 2 (high confidence in the right answer):**
```
Same codebook, model assigns probability 0.9 to the correct token.

L_CE = -log(0.9) ≈ 0.105   ← a low loss, the model was confidently right
```

**Beginner question:** *What if the model is confidently WRONG — high
probability on the wrong token?* Then the probability assigned to the
*correct* token would be very small (since probabilities across all
tokens sum to 1), producing a large `-log(small number)` loss — cross-
entropy punishes confident wrongness harshly, which is exactly the
behavior you want during training.

---

## 11. GE2E Loss (Generalized End-to-End, Speaker Embeddings)

**Definition:** A loss that trains the speaker encoder (Phase 8) to
cluster same-speaker recordings close together in embedding space, and
push different speakers' embeddings apart.

**How to read it:**
```
S(i, j) = cos(e_i, c_j) · w + b
L_GE2E = -Σ_i  log( e^(S(i, i_true)) / Σ_j e^(S(i, j)) )
```
Say it as: "measure how similar this utterance's embedding is to every
speaker's average embedding (centroid) in the batch, then use a softmax-
style cross-entropy to reward it being most similar to its *own* true
speaker's centroid."

**Why we use it:** This is what makes zero-shot voice cloning (Phase 8)
actually work — without this loss, the speaker encoder would have no
training signal pushing it to represent "who is speaking" consistently
across different recordings of the same person.

**Formula:**
```
S(i, j) = cos(e_i, c_j) · w + b
L_GE2E = -log( e^(S(i, i_true)) / Σ_j e^(S(i, j)) )   (for one utterance i)
```

**Example 1 (confidently correct clustering, ignoring w,b for simplicity):**
```
3 speakers in the batch. Utterance i belongs to speaker 1.

S(i,1) = 0.95   (very close to its own speaker's centroid)
S(i,2) = 0.10
S(i,3) = 0.10

numerator = e^0.95 ≈ 2.586
denominator = e^0.95 + e^0.10 + e^0.10 ≈ 2.586+1.105+1.105 = 4.796

L_GE2E = -log(2.586/4.796) = -log(0.539) ≈ 0.618
```

**Example 2 (ambiguous clustering — equally close to all speakers):**
```
S(i,1) = 0.5, S(i,2) = 0.5, S(i,3) = 0.5   (no clear preference)

numerator = e^0.5 ≈ 1.649
denominator = 3 × e^0.5 ≈ 4.946

L_GE2E = -log(1.649/4.946) = -log(0.333) ≈ 1.099

Notice: this is a bigger loss than Example 1 — the embedding genuinely
isn't distinguishing this speaker from the others yet, which is exactly
what a badly-trained (or early in training) speaker encoder would produce.
```

**Beginner question:** *What's a "centroid" in plain terms?* Just the
average embedding across every utterance belonging to one speaker in the
current training batch — a rough "typical position" for that speaker in
the embedding space, recomputed fresh for every batch as training
progresses.

---

*Continue to Part 2 for the math behind duration/pitch/loudness losses,
diffusion/flow-matching, GRPO, DPO, gradient orthogonalization,
watermarking, and quantisation.*
