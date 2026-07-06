# SELF_LEARNING_VOICE_STUDIO.md — `aarambh-voice-selflearn`

> Companion to ARCHITECTURE_VOICE_STUDIO.md Part 1/2. This is the full
> design for online self-learning — the piece the draft architecture was
> missing entirely. Mirrors the design discipline behind Manas (associative
> memory, gradient orthogonalization anti-forgetting, self-growing
> architecture, confidence-gated commits), re-derived for audio.

---

## Table of Contents

1. [Why This Exists](#1-why-this-exists)
2. [Design Principles](#2-design-principles)
3. [Architecture Overview](#3-architecture-overview)
4. [Associative Memory for Voices & Styles](#4-associative-memory-for-voices--styles)
5. [Anti-Forgetting via Gradient Orthogonalization](#5-anti-forgetting-via-gradient-orthogonalization)
6. [Self-Growing Adapter Bank](#6-self-growing-adapter-bank)
7. [Confidence-Gated Commit Loop](#7-confidence-gated-commit-loop)
8. [Crate Structure](#8-crate-structure)
9. [Data Structures](#9-data-structures)
10. [Training / Update Algorithm — Full Walkthrough](#10-training--update-algorithm--full-walkthrough)
11. [Integration With `-serve` (Online Path)](#11-integration-with--serve-online-path)
12. [Integration With `-finetune` and `-align` (Offline Paths)](#12-integration-with--finetune-and--align-offline-paths)
13. [Failure Modes & Rollback](#13-failure-modes--rollback)
14. [Hardware Budget (i3-only)](#14-hardware-budget-i3-only)
15. [Tests](#15-tests)
16. [Milestone](#16-milestone)

---

## 1. Why This Exists

Without this crate, every new speaker, accent, singing style, or user
correction requires a full `aarambh-voice-finetune` job — a deliberate,
offline, Kaggle-scale process (§20, Part 2). That's the right tool for
*big* adaptations. It's the wrong tool for "here's 8 seconds of a new
voice, remember it" or "that clone sounded a bit nasal, here's a
corrected sample."

Manas solved the equivalent problem for text/knowledge by combining
associative memory (store-and-retrieve rather than re-train-from-scratch),
gradient orthogonalization (so learning fact #50 doesn't erase fact #1),
and a self-growing architecture (capacity added incrementally, not
pre-allocated). This document re-derives the same three mechanisms for
audio, where the equivalent unit isn't a "fact" — it's a speaker identity,
a singing style, or a correction to one.

## 2. Design Principles

| Principle (from Manas) | Applied here |
|---|---|
| Store, don't retrain, for retrieval-shaped problems | New voices are stored as embeddings + small adapter deltas, not new base-model weights |
| Anti-forgetting must be structural, not hopeful | Gradient orthogonalization is enforced at update time, not just monitored after the fact |
| Growth is incremental | Each new speaker/style adds a small, bounded adapter — the system does not need to know its final size in advance |
| Confidence gating before any commit | An update is provisional until it passes an eval check; rejected updates leave the deployed model untouched |
| i3-first | Every mechanism here must run inference-time-cheap on an 8GB laptop — this is not a Kaggle-only crate |

## 3. Architecture Overview

```
                     ┌─────────────────────────────┐
                     │   Inference request arrives   │
                     │   (new voice sample, or a      │
                     │    correction to an existing    │
                     │    one)                          │
                     └───────────────┬─────────────────┘
                                     │
                     ┌───────────────┴─────────────────┐
                     │  4. Associative Memory Lookup      │
                     │     — is this speaker/style already │
                     │       known? retrieve embedding      │
                     └───────────────┬─────────────────┘
                              known? │ not known / correction?
                     ┌───────────────┴─────────────────┐
                     │  6. Self-Growing Adapter Bank        │
                     │     — allocate or update a small      │
                     │       LoRA-style delta for this        │
                     │       speaker/style                     │
                     └───────────────┬─────────────────┘
                                     │
                     ┌───────────────┴─────────────────┐
                     │  5. Gradient Orthogonalization        │
                     │     — project the update's gradient    │
                     │       away from directions used by      │
                     │       existing speakers/styles            │
                     └───────────────┬─────────────────┘
                                     │
                     ┌───────────────┴─────────────────┐
                     │  7. Confidence-Gated Commit           │
                     │     — run -eval on a small held-out     │
                     │       set; commit only if no regression   │
                     └───────────────┬─────────────────┘
                              pass   │   fail
                     ┌───────────────┴───┐   ┌──────────────┐
                     │ Commit to adapter    │   │ Roll back,     │
                     │ bank + memory index   │   │ log, fall back  │
                     │                         │   │ to base voice   │
                     └───────────────────────┘   └──────────────┘
```

## 4. Associative Memory for Voices & Styles

A key-value store, keyed by a compact identity embedding (the same
256-dim speaker embedding space used in Part 1 §8), valued by:
- the adapter delta ID (pointer into the self-growing adapter bank, §6)
- provenance metadata (when learned, from how many samples, consent record)
- a running confidence score (updated every time this entry is used and
  re-validated)

Retrieval at inference time is nearest-neighbour lookup in embedding
space, not exact-match — this is what lets "a slightly different sample
of a voice we already know" resolve to the existing adapter instead of
spuriously allocating a new one. Threshold-tuned (cosine similarity ≥ 0.85
by default) so genuinely new voices still get a new entry rather than
being incorrectly merged into a similar-sounding existing one.

```rust
pub struct MemoryEntry {
    pub identity_embedding: [f32; 256],
    pub adapter_id: AdapterId,
    pub provenance: Provenance,      // timestamp, sample_count, consent_ref
    pub confidence: f32,             // 0.0-1.0, updated on each re-validation
}

pub struct AssociativeMemory {
    entries: Vec<MemoryEntry>,       // small enough to be linear-scanned
                                      // at i3 scale; upgrade to an ANN index
                                      // (e.g. HNSW) only if entry count grows
                                      // past a few thousand
}

impl AssociativeMemory {
    pub fn lookup(&self, query_embedding: &[f32; 256]) -> Option<&MemoryEntry>;
    pub fn insert(&mut self, entry: MemoryEntry);
    pub fn update_confidence(&mut self, adapter_id: AdapterId, new_confidence: f32);
}
```

## 5. Anti-Forgetting via Gradient Orthogonalization

The core mechanism, ported directly from Manas's approach: when computing
the gradient update for a new or corrected speaker/style adapter, project
that gradient to be orthogonal to the subspace spanned by gradients used
for *existing* adapters, so the update cannot degrade them.

```
Let G_existing = [g_1, g_2, ..., g_n]   (stacked gradient directions used
                                          by the n existing adapters,
                                          cached at commit time — not
                                          recomputed from scratch each time)

Let g_new = gradient of the new/corrected update

g_new_orthogonal = g_new - G_existing @ (G_existing^+ @ g_new)
                    (G_existing^+ = pseudo-inverse, or a cheaper
                     Gram-Schmidt projection when n is small, which it
                     always is at adapter-delta scale — this is not a
                     full-model gradient, so the projection is cheap)

Apply g_new_orthogonal, not g_new, to the new adapter delta.
```

Because each adapter is a small LoRA-style delta (not the full model),
`G_existing` stays small and this projection is cheap enough to run on
the i3 at update time — this would be prohibitively expensive if applied
to full fine-tuning gradients, which is exactly why self-learning operates
on adapter deltas rather than the base model.

## 6. Self-Growing Adapter Bank

Each new speaker/style gets a small LoRA delta (rank 4–8, targeting the
speaker-conditioning injection points from Part 1 §8.1 and, for styles,
the singing-melody cross-attention points from Part 1 §13.1) — allocated
on demand, not pre-provisioned. The bank grows one entry at a time; there
is no fixed ceiling baked into the architecture (matches Manas's
"self-growing" framing directly), though a practical soft cap (e.g.
prune/merge adapters with confidence below a floor, oldest-first) keeps
memory bounded on an 8GB machine — see §14.

```rust
pub struct AdapterDelta {
    pub id: AdapterId,
    pub rank: usize,               // 4-8
    pub target_modules: Vec<ModulePath>,
    pub weights: LoraWeights,       // A, B low-rank matrices
}

pub struct AdapterBank {
    deltas: HashMap<AdapterId, AdapterDelta>,
    memory_budget_bytes: usize,     // enforced soft cap, see §14
}

impl AdapterBank {
    pub fn allocate(&mut self, target_modules: Vec<ModulePath>) -> AdapterId;
    pub fn update(&mut self, id: AdapterId, orthogonalized_gradient: &Gradient, lr: f32);
    pub fn prune_lowest_confidence(&mut self, memory: &AssociativeMemory);
}
```

## 7. Confidence-Gated Commit Loop

No update is trusted by default. Every candidate update — new speaker,
new style, or a correction — goes through:

```
1. Compute orthogonalized update (§5), apply to a *staged copy* of the
   relevant adapter delta (never mutate the live, deployed delta directly)
2. Run a small subset of aarambh-voice-eval metrics (§24, Part 2) most
   relevant to what changed:
     - new/corrected speaker → speaker-similarity + ASR-roundtrip WER
     - new/corrected singing style → pitch-accuracy + MOS-proxy
3. Compare staged-copy scores against the pre-update scores on the same
   fixed small held-out prompt set
4. If staged scores are >= pre-update scores (within a small tolerance,
   e.g. -0.02 absolute, to avoid rejecting noise-level fluctuation):
     commit staged copy → live adapter bank, update AssociativeMemory
     confidence upward
   else:
     discard staged copy, log the rejection with the eval delta, leave
     live adapter untouched, do NOT silently retry — surface the
     rejection so the person providing the sample knows it didn't help
```

This loop is what makes self-learning safe to run unattended at
inference time — a bad sample (noisy audio, mislabeled speaker, a
correction that actually makes things worse) cannot silently degrade a
voice that's already working.

## 8. Crate Structure

```
crates/aarambh-voice-selflearn/
├── src/
│   ├── memory.rs        # AssociativeMemory (§4)
│   ├── orthogonalize.rs # Gradient orthogonalization (§5)
│   ├── adapter_bank.rs  # AdapterBank, self-growing (§6)
│   ├── commit.rs        # Confidence-gated commit loop (§7)
│   ├── update.rs        # Full update algorithm (§10), entry point called by -serve
│   └── lib.rs
```

## 9. Data Structures

(See code blocks in §4 and §6 above — `MemoryEntry`, `AssociativeMemory`,
`AdapterDelta`, `AdapterBank` are the complete public surface. `update.rs`
composes these into the single entry point described next.)

## 10. Training / Update Algorithm — Full Walkthrough

```rust
pub fn online_update(
    memory: &mut AssociativeMemory,
    bank: &mut AdapterBank,
    sample: AudioSample,           // new/corrected speaker or style sample
    identity_embedding: [f32; 256],
    eval_harness: &EvalHarness,     // from aarambh-voice-eval
) -> UpdateResult {
    // 1. Lookup — known speaker/style, or new?
    let existing = memory.lookup(&identity_embedding);

    // 2. Allocate or reuse adapter
    let adapter_id = existing
        .map(|e| e.adapter_id)
        .unwrap_or_else(|| bank.allocate(default_target_modules()));

    // 3. Compute raw gradient from the sample (small, single-sample or
    //    few-shot batch — this is NOT a full training epoch)
    let raw_gradient = compute_gradient(&sample, adapter_id, bank);

    // 4. Orthogonalize against existing adapters (§5)
    let existing_gradients = bank.cached_gradient_directions();
    let orthogonal_gradient = orthogonalize(raw_gradient, &existing_gradients);

    // 5. Stage the update — never mutate live weights directly
    let staged = bank.stage_update(adapter_id, &orthogonal_gradient, lr = 1e-3);

    // 6. Confidence gate (§7)
    let pre_scores = eval_harness.score_relevant_metrics(adapter_id, bank);
    let post_scores = eval_harness.score_relevant_metrics_staged(&staged);

    if post_scores.meets_or_exceeds(&pre_scores, tolerance = 0.02) {
        bank.commit(staged);
        memory.insert_or_update(identity_embedding, adapter_id, confidence = post_scores.overall());
        UpdateResult::Committed { adapter_id, delta_scores: post_scores - pre_scores }
    } else {
        UpdateResult::Rejected { reason: post_scores.regression_summary(&pre_scores) }
    }
}
```

## 11. Integration With `-serve` (Online Path)

`aarambh-voice-serve` calls `online_update()` when a request includes a
`learn_from_this: true` flag alongside a voice/style sample and valid
consent (§23, Part 2 — self-learning does not bypass the safety layer;
consent gating applies identically to online and offline paths). The
update runs asynchronously after the response is returned, so it never
adds latency to the request that triggered it.

## 12. Integration With `-finetune` and `-align` (Offline Paths)

Self-learning and offline fine-tuning are complementary, not competing:

| | `-finetune` (§20) | `-selflearn` (this doc) |
|---|---|---|
| Trigger | Deliberate, scheduled job | Inference-time, per-request |
| Scale | Full LoRA/QLoRA/DoRA rank, dataset-sized | Small delta, few-shot |
| Hardware | Kaggle | i3 |
| Use case | "Adapt to this speaker properly, with a real dataset" | "Remember this voice from one sample" or "fix this one thing" |
| Safety | Reviewed before deploy (human in the loop) | Confidence-gated automatically (§7) |

Periodically, high-confidence, frequently-used self-learned adapters are
good *candidates* to graduate into a proper `-finetune` job (more data,
full rank) — this is a manual decision, not automated, since it's the
point where a human should look at what the system has learned before
investing Kaggle time in it.

## 13. Failure Modes & Rollback

| Failure | Handling |
|---|---|
| Noisy/corrupted input sample | Update is computed and staged as normal; confidence gate (§7) rejects it if it regresses eval scores — no special-casing needed, the gate is the safety net |
| Two very similar but distinct speakers merged incorrectly | Similarity threshold (§4) tuned conservatively (0.85); if a false merge still occurs, provenance metadata lets it be manually split and re-keyed |
| Adapter bank grows unbounded | Soft memory cap (§14) with confidence-based pruning, oldest-lowest-confidence-first |
| Orthogonalization projection numerically unstable as `n` (existing adapters) grows large | Cap `G_existing` to the K most-relevant existing adapters (nearest in embedding space to the new one) rather than all of them — bounds the pseudo-inverse cost regardless of total bank size |

## 14. Hardware Budget (i3-only)

Self-learning is designed to never require Kaggle:

| Operation | Approx. cost on i3 |
|---|---|
| Associative memory lookup (linear scan, <5k entries) | <10ms |
| Single adapter gradient computation (few-shot, rank 4-8) | ~1-3s |
| Orthogonalization projection (K≤8 existing adapters) | <100ms |
| Confidence-gate eval subset (2-3 metrics, small held-out set) | ~5-15s |
| **Total per online update** | **~10-20 seconds**, async, non-blocking |

Adapter bank soft cap: default 200MB (roughly a few hundred rank-8 deltas
at Small/Medium scale) — comfortably inside the 8GB budget alongside a
running inference session (see Part 2 §27 for the base memory table this
adds on top of).

## 15. Tests

```
[ ] AssociativeMemory::lookup returns the correct entry for a query
    embedding within similarity threshold, and None otherwise
[ ] orthogonalize() produces a gradient with zero (within float tolerance)
    projection onto each existing adapter's cached gradient direction
[ ] AdapterBank::allocate never collides IDs; prune_lowest_confidence
    respects the memory budget
[ ] online_update() commits when staged eval scores meet/exceed
    pre-update scores, and rejects (leaving live weights untouched) when
    they don't
[ ] A sequence of 50 online updates for 50 distinct speakers does not
    degrade speaker-similarity score for speaker #1, measured after
    speaker #50's update (the direct anti-forgetting regression test)
[ ] Rejected updates never mutate the live adapter bank (staged copy is
    discarded, not partially applied)
```

## 16. Milestone

```
aarambh-voice-studio learn --sample new_voice.wav --identity-hint "warm, mid-30s"
```
produces either a `Committed` result (new memory entry + adapter created,
confidence score printed) or a `Rejected` result (reason printed, live
model unchanged) — and the 50-speaker anti-forgetting regression test
(§15) passes. Tag: `v0.1.0-selflearn`.
