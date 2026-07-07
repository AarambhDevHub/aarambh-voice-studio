# aarambh-voice-studio: The Complete Math Formula Guide (Part 2 of 2)

### Continues directly from Part 1, formula 11 (GE2E Loss)

Same format as Part 1, for every formula:
- **What it's called** (definition)
- **How to read the symbols**
- **Why we use it in aarambh-voice-studio**
- **The formula itself**
- **2 fully solved examples**
- **A beginner question**

---

## 12. Duration Loss (L2)

**Definition:** A loss that measures how far off the model's predicted
duration for each phoneme is from the true, forced-aligned duration.

**How to read it:**
```
L_dur = (1/N) · Σ_i  (d_i - d̂_i)²
```
Say it as: "for each of the N phonemes, subtract the predicted duration
from the true duration, square it, then average over all of them."

**Why we use it:** This trains the duration-prediction head used in
Phase 6's TTS baseline and Phase 13's singing model — without it, the
model would have no explicit signal for "how long should this sound
last," and would need to guess that purely from the main token-prediction
loss, which is a much weaker signal for timing specifically.

**Formula:**
```
L_dur = (1/N) · Σ_i  (d_i - d̂_i)²
```

**Example 1 (small, evenly-distributed errors):**
```
True durations (frames):      [4, 6, 3, 8, 5]
Predicted durations (frames): [5, 6, 2, 8, 6]

squared errors: (4-5)²=1, (6-6)²=0, (3-2)²=1, (8-8)²=0, (5-6)²=1
L_dur = (1+0+1+0+1) / 5 = 3/5 = 0.6
```

**Example 2 (one large error dominates):**
```
True durations:      [4, 6, 3, 8, 5]
Predicted durations: [4, 6, 3, 8, 15]   ← one badly wrong prediction

squared errors: 0, 0, 0, 0, (5-15)²=100
L_dur = (0+0+0+0+100) / 5 = 100/5 = 20.0

Notice how much bigger this loss is compared to Example 1 — squaring
means one large error contributes far more than several small ones,
which pushes training to prioritize fixing big mistakes first.
```

**Beginner question:** *Why square the error instead of just using the
plain difference?* Squaring makes the loss always positive (so errors in
either direction, too long or too short, count the same way) and
penalizes large errors disproportionately more than small ones — a
model that's occasionally very wrong is penalized more than one that's
consistently slightly off.

---

## 13. Pitch (F0) Loss

**Definition:** The same kind of loss as duration, applied to fundamental
frequency (F0) — how high or low the voice sounds at each moment.

**How to read it:**
```
L_pitch = (1/N) · Σ_i  (f0_i - f̂0_i)²
```
Say it as: identical structure to duration loss, but comparing true vs.
predicted pitch values (usually in log-Hz, for reasons explained below)
instead of durations.

**Why we use it:** This is what makes Phase 13's singing model actually
hit the correct musical notes specified by the input melody, rather than
just producing intelligible lyrics at whatever pitch feels natural.

**Formula:**
```
L_pitch = (1/N) · Σ_i  (f0_i - f̂0_i)²    (typically in log-Hz space)
```

**Example 1 (close predictions):**
```
True log-F0:      [4.8, 4.9, 5.0]   (roughly 121Hz, 134Hz, 148Hz)
Predicted log-F0: [4.7, 4.9, 5.1]

squared errors: (4.8-4.7)²=0.01, (4.9-4.9)²=0.0, (5.0-5.1)²=0.01
L_pitch = (0.01+0.0+0.01)/3 ≈ 0.0067
```

**Example 2 (why log-Hz matters — a raw Hz comparison would be misleading):**
```
Low true pitch: 100Hz, predicted 110Hz  → raw Hz error = 10
High true pitch: 800Hz, predicted 810Hz → raw Hz error = 10 (same!)

But perceptually, a 10Hz shift at 100Hz is a much bigger relative
change (10%) than a 10Hz shift at 800Hz (1.25%) — computing the loss
in log-Hz space instead of raw Hz makes the loss reflect this
relative, perceptual difference rather than treating both cases
as "equally wrong."
```

**Beginner question:** *Why does singing need this but ordinary speech
(Phase 6) doesn't have an explicit pitch loss?* Ordinary speech's natural
pitch variation (intonation) is learned implicitly as part of sounding
natural — there's no single "correct" pitch for a spoken sentence. Sung
lyrics, by contrast, have an externally-specified correct pitch (the
melody), so there's a real ground truth to train against explicitly.

---

## 14. LUFS Loudness Matching

**Definition:** A standardized way of measuring how loud audio sounds to
a human, used to match a final mixed song to a target loudness level.

**How to read it (simplified):**
```
LUFS = -0.691 + 10·log10( Σ_c Σ_t  G_c · z_c[t]² )
L_loud = (LUFS_target - LUFS_measured)²
```
Say it as: "square and weight the audio signal to match human hearing
sensitivity across frequencies, average it over time and channels, then
convert to a logarithmic loudness scale — and penalize the mix for how
far its measured loudness is from the target."

**Why we use it:** Simple amplitude-based loudness comparisons don't
match how loud something *actually sounds* to a listener — LUFS accounts
for the fact that human loudness perception depends heavily on which
frequencies are present, not just the raw signal size. Phase 15's mixing
model uses this to make sure a finished song matches typical streaming-
platform loudness (commonly -14 LUFS), rather than sounding too quiet or
too loud next to other music.

**Formula:**
```
L_loud = (LUFS_target - LUFS_measured)²
```

**Example 1 (mix is too quiet):**
```
LUFS_target = -14,  LUFS_measured = -18

L_loud = (-14 - (-18))² = (4)² = 16   ← significant loss, needs more gain
```

**Example 2 (mix is very close to target):**
```
LUFS_target = -14,  LUFS_measured = -14.3

L_loud = (-14 - (-14.3))² = (0.3)² = 0.09   ← tiny loss, nearly there
```

**Beginner question:** *Why -14 LUFS specifically?* It's a widely-used
target across major streaming platforms — matching it means a generated
song won't sound jarringly different in loudness when played in a
playlist alongside professionally mastered music; it's a practical
industry convention this project adopts rather than deriving from first
principles.

---

## 15. Diffusion / Flow-Matching Objective (Singing Stage B)

**Definition:** The training objective for Phase 14's optional
refinement pass — the model learns to predict what random noise was
added to clean audio, so it can later reverse that process starting from
pure noise.

**How to read it:**
```
L_diffusion = E_t [ || ε - ε_θ(x_t, t, cond) ||² ]
where x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε,   ε ~ N(0, I)
```
Say it as: "take clean audio x_0, mix in a random amount of noise ε
according to a schedule ᾱ_t, and train the model to guess exactly what
noise ε was added, given the noisy result, the noise level t, and
Stage A's conditioning."

**Why we use it:** This is the mechanism behind Phase 14's naturalness
improvement over Phase 13's pure autoregressive output — training the
model to "undo" noise, then running that process backward at generation
time (starting from pure noise and gradually cleaning it up), tends to
produce more natural results for singing specifically.

**Formula:**
```
L_diffusion = E_t [ || ε - ε_θ(x_t, t, cond) ||² ]
```

**Example 1 (close noise prediction):**
```
true added noise ε = 0.42
model's predicted noise ε_θ = 0.39

squared error = (0.42 - 0.39)² = (0.03)² = 0.0009   ← small loss
```

**Example 2 (poor noise prediction, early in training):**
```
true added noise ε = 0.42
model's predicted noise ε_θ = -0.10   (way off, even wrong sign)

squared error = (0.42 - (-0.10))² = (0.52)² = 0.2704   ← much bigger loss

As training progresses, ε_θ's predictions should look more like
Example 1 across a wide range of noise levels t and conditioning
inputs, not just for one specific case.
```

**Beginner question:** *How does "predicting noise" turn into "generating
new singing"?* At generation time, the process runs in reverse: start
from pure random noise (as if `t` were at its maximum), ask the model
"what noise would you remove here?", subtract a bit of that predicted
noise, and repeat many times — each step revealing a little more of a
clean result, conditioned throughout on Stage A's output.

---

## 16. GRPO — Group-Relative Advantage

**Definition:** A reinforcement-learning method that samples several
candidate outputs for the same prompt, scores them, and reinforces
whichever candidates scored better than their own group's average.

**How to read it:**
```
A_i = R_i - mean(R_1, ..., R_K)
L_GRPO = -E_i [ A_i · log π_θ(sample_i) ]
```
Say it as: "for each of the K sampled candidates, subtract the group's
average reward from this candidate's reward to get its advantage — then
push the model to increase the probability of high-advantage candidates
and decrease the probability of low-advantage ones."

**Why we use it:** This is Phase 22's mechanism for directly optimizing
generation quality (naturalness, speaker fidelity, emotion accuracy)
rather than just matching training recordings — and doing it without
needing a separate, expensive-to-train "value network" the way an older
method like PPO would require.

**Formula:**
```
A_i = R_i - mean(R_1, ..., R_K)
```

**Example 1 (4 candidates, clear winner and loser):**
```
Rewards: R = [0.9, 0.6, 0.5, 0.4]

mean(R) = (0.9+0.6+0.5+0.4)/4 = 2.4/4 = 0.6

A_1 = 0.9 - 0.6 = +0.3   (reinforce — well above average)
A_2 = 0.6 - 0.6 =  0.0   (neutral — exactly average)
A_3 = 0.5 - 0.6 = -0.1   (discourage slightly)
A_4 = 0.4 - 0.6 = -0.2   (discourage more)
```

**Example 2 (all candidates roughly equal — little signal):**
```
Rewards: R = [0.61, 0.60, 0.59, 0.60]

mean(R) = (0.61+0.60+0.59+0.60)/4 = 2.40/4 = 0.60

A_1 = +0.01, A_2 = 0.00, A_3 = -0.01, A_4 = 0.00

Notice: when all candidates score almost identically, the advantages
are tiny — GRPO naturally produces little training signal in cases
where the model already generates consistently similar-quality output,
and a much stronger signal when quality varies a lot between samples.
```

**Beginner question:** *Why compare against the group's own average
instead of some fixed target score?* Using the group's own average as
the baseline means GRPO doesn't need a separately-trained model to
predict "what reward should I expect here" (a value network) — the other
sampled candidates for the same prompt serve as a free, built-in point of
comparison.

---

## 17. DPO — Direct Preference Optimization

**Definition:** A cheaper alternative to GRPO that builds (better, worse)
example pairs ahead of time and trains the model to prefer the better one
— no live sampling loop needed during training.

**How to read it:**
```
L_DPO = -log σ( β · [ (log π_θ(y_w|x) - log π_ref(y_w|x))
                     - (log π_θ(y_l|x) - log π_ref(y_l|x)) ] )
```
Say it as: "compare how much more the current model prefers the winner
(y_w) than the frozen reference model does, against how much more it
prefers the loser (y_l) than the reference does — and push that gap as
wide as possible."

**Why we use it:** This is Phase 22's cheaper complement to GRPO —
because the (winner, loser) pairs are built once, offline, from
candidates already scored by the evaluation harness, training doesn't
need a live generation loop, which is significantly less compute-
intensive than GRPO's approach.

**Formula:**
```
L_DPO = -log σ( β · [ (log π_θ(y_w) - log π_ref(y_w))
                     - (log π_θ(y_l) - log π_ref(y_l)) ] )
```

**Example 1 (model already prefers the winner more than the reference did):**
```
log π_θ(y_w) = -2.0,  log π_ref(y_w) = -2.2
log π_θ(y_l) = -3.0,  log π_ref(y_l) = -2.5
β = 0.1

term_w = -2.0 - (-2.2) = 0.2
term_l = -3.0 - (-2.5) = -0.5
inner = 0.1 × (0.2 - (-0.5)) = 0.1 × 0.7 = 0.07

L_DPO = -log(σ(0.07)) = -log(0.5175) ≈ 0.659
```

**Example 2 (model hasn't learned the preference yet — no gap):**
```
log π_θ(y_w) = -2.2,  log π_ref(y_w) = -2.2   (identical to reference)
log π_θ(y_l) = -2.5,  log π_ref(y_l) = -2.5   (identical to reference)
β = 0.1

term_w = -2.2 - (-2.2) = 0.0
term_l = -2.5 - (-2.5) = 0.0
inner = 0.1 × (0.0 - 0.0) = 0.0

L_DPO = -log(σ(0.0)) = -log(0.5) ≈ 0.693

This is the "starting point" loss — before any training has shifted
the model's preferences at all, σ(0) = 0.5 exactly, giving this fixed
loss value that training then works to reduce.
```

**Beginner question:** *What is π_ref for, exactly?* It's a frozen copy
of the model *before* this alignment stage began — it acts as an anchor,
so the loss measures "how much more has the model shifted toward the
winner compared to where it started," rather than measuring raw
preference in isolation, which helps keep training from drifting too far,
too fast, from a model that already works reasonably well.

---

## 18. Gradient Orthogonalization (Self-Learning Anti-Forgetting)

**Definition:** A mathematical technique that adjusts a new training
update so it has zero overlap with the directions used by
already-learned updates — mathematically guaranteeing the new update
can't disturb what's already been learned.

**How to read it:**
```
g_new_orthogonal = g_new - G · (G⁺ · g_new)
```
Say it as: "take the new update's raw gradient, figure out how much of
it overlaps with the space of directions existing adapters rely on
(`G · (G⁺ · g_new)`), and subtract that overlapping part away — what's
left is safe to apply."

**Why we use it:** This is the core safety mechanism behind Phase 23's
self-learning — it's what lets the system learn a 51st new voice without
mathematically risking any damage to the 50 voices it already knows,
rather than just hoping in practice that a small update won't cause
noticeable forgetting.

**Formula (2D simplified case, one existing direction):**
```
projection of g_new onto g_old = (g_new · g_old) × g_old
g_new_orthogonal = g_new - projection
```

**Example 1 (partial overlap):**
```
existing adapter's direction: g_old = [1, 0]   (already normalized)
new update's raw gradient:    g_new = [0.6, 0.8]

projection = (g_new · g_old) × g_old = (0.6×1 + 0.8×0) × [1,0]
           = 0.6 × [1,0] = [0.6, 0]

g_new_orthogonal = [0.6, 0.8] - [0.6, 0] = [0, 0.8]

The resulting update only changes the dimension the old adapter
doesn't use at all — the old adapter's behavior is unaffected.
```

**Example 2 (no overlap to begin with):**
```
existing adapter's direction: g_old = [1, 0]
new update's raw gradient:    g_new = [0, 0.9]   (already orthogonal)

projection = (g_new · g_old) × g_old = (0×1 + 0.9×0) × [1,0] = [0, 0]

g_new_orthogonal = [0, 0.9] - [0, 0] = [0, 0.9]   ← unchanged

When the new update was already independent of the existing adapter,
orthogonalization has nothing to remove — this shows the mechanism
only intervenes exactly as much as needed, never more.
```

**Beginner question:** *Does this mean the new update is always weaker
than it "wants" to be?* Only along the specific directions existing
adapters depend on — everywhere else, the update proceeds at full
strength. The tradeoff is deliberate: a small amount of potential update
strength is sacrificed specifically to guarantee no forgetting, which is
judged worth it for a system meant to run unattended in production.

---

## 19. Watermarking (Spread-Spectrum, Conceptual)

**Definition:** A technique for hiding an inaudible, detectable signal
inside generated audio, so it can later be identified as AI-generated
even without any visible label.

**How to read it (embedding):**
```
x_watermarked[n] = x[n] + α · w[n]
```
Say it as: "add a tiny amount (scaled by α) of a special pseudo-random
pattern w[n] to every sample of the generated audio."

**How to read it (detection):**
```
correlation = Σ_n  x_watermarked[n] · w[n]
```
Say it as: "multiply the (possibly watermarked) audio against the same
known pattern, sample by sample, and sum the results — a real match
produces a measurably higher correlation than chance."

**Why we use it:** This is Phase 19's mechanism for making generated
audio traceable after the fact, supporting responsible use of the
project's cloning and singing capabilities — without needing to strip out
or degrade the audible content at all.

**Formula:**
```
x_watermarked[n] = x[n] + α · w[n]
correlation = Σ_n  x_watermarked[n] · w[n]
```

**Example 1 (embedding, a few samples):**
```
x = [0.10, 0.20, -0.05],  w = [0.01, -0.01, 0.01],  α = 0.02

x_watermarked = [0.10 + 0.02×0.01, 0.20 + 0.02×(-0.01), -0.05 + 0.02×0.01]
              = [0.1002, 0.1998, -0.0498]

Notice: the change (0.0002, -0.0002, 0.0002) is tiny relative to
the original values — this is what keeps the watermark inaudible.
```

**Example 2 (detection — correlation on watermarked vs. clean audio):**
```
w = [0.01, -0.01, 0.01]

Watermarked audio (from Example 1): [0.1002, 0.1998, -0.0498]
correlation = 0.1002×0.01 + 0.1998×(-0.01) + (-0.0498)×0.01
            = 0.001002 - 0.001998 - 0.000498 = -0.001494

Clean (non-watermarked) audio: [0.10, 0.20, -0.05]  (original x)
correlation = 0.10×0.01 + 0.20×(-0.01) + (-0.05)×0.01
            = 0.001 - 0.002 - 0.0005 = -0.0015

At this tiny 3-sample scale the difference looks small, but over a
full-length clip (thousands of samples), the watermarked audio's
correlation consistently and measurably exceeds what random chance
would produce for unwatermarked audio — that gap, checked against a
calibrated threshold, is what the detector actually uses.
```

**Beginner question:** *If the change is so tiny, how is it reliably
detectable?* Because the pattern `w[n]` is spread across many samples and
known exactly by the detector — even though each individual sample's
change is minuscule, summing the correlation across a whole clip
accumulates a statistically reliable signal, the same principle behind
spread-spectrum techniques used elsewhere in signal processing.

---

## 20. Quantisation (INT8 / INT4)

**Definition:** A technique for compressing a model's full-precision
weight values down to a small number of discrete levels (e.g. 16 levels
for 4-bit), trading a small amount of precision for a large reduction in
memory.

**How to read it:**
```
q = round( (x - x_min) / (x_max - x_min) × (2^b - 1) )
x_dequantized = q / (2^b - 1) × (x_max - x_min) + x_min
```
Say it as: "figure out where x falls, proportionally, between x_min and
x_max, scale that proportion up to fit into however many discrete levels
b bits allow, and round to the nearest one — then, to use the value
again, reverse the process to get an approximation of the original."

**Why we use it:** This is Phase 20's mechanism for fitting larger models
into the memory available on an ordinary laptop — INT4 quantisation
shrinks a model to roughly 1/8th its full-precision size, at a small,
measured quality cost checked against the evaluation harness.

**Formula:**
```
q = round( (x - x_min) / (x_max - x_min) × (2^b - 1) )
x_dequantized = q / (2^b - 1) × (x_max - x_min) + x_min
```

**Example 1 (INT4 quantisation of a mid-range value):**
```
x_min = -2.0, x_max = 2.0, b = 4 (so 2^4 - 1 = 15 levels), x = 0.5

q = round( (0.5 - (-2.0)) / (2.0 - (-2.0)) × 15 )
  = round( (2.5/4.0) × 15 ) = round(0.625 × 15) = round(9.375) = 9

x_dequantized = 9/15 × 4.0 + (-2.0) = 0.6×4.0 - 2.0 = 2.4 - 2.0 = 0.4

Original 0.5 becomes 0.4 after the round-trip — a small, expected
quantization error.
```

**Example 2 (INT8 quantisation of the same value — more precision available):**
```
x_min = -2.0, x_max = 2.0, b = 8 (so 2^8 - 1 = 255 levels), x = 0.5

q = round( (0.5 - (-2.0)) / (2.0 - (-2.0)) × 255 )
  = round( 0.625 × 255 ) = round(159.375) = 159

x_dequantized = 159/255 × 4.0 + (-2.0) = 0.6235×4.0 - 2.0
              = 2.494 - 2.0 = 0.494

Original 0.5 becomes 0.494 — much closer than INT4's 0.4, because
255 discrete levels give far finer resolution than 15 do.
```

**Beginner question:** *Why not always use INT8 then, if it's more
accurate?* INT8 uses twice the memory of INT4 (8 bits vs 4 bits per
weight) — the choice is a genuine tradeoff between accuracy and memory
footprint, and Phase 20's evaluation step exists specifically to check
whether INT4's extra error is small enough to accept for a given use
case, rather than assuming one setting is always right.

---

*That covers the full math behind every loss function and training
algorithm referenced in the Complete Guide. For where the actual training
data for each of these comes from, see
`VOICE_STUDIO_AUDIO_ML_TERMINOLOGY_AND_DATASET_GUIDE.md`.*
