# Contributing to aarambh-voice-studio

Thank you for taking the time to contribute. Every bug report, documentation improvement, test, benchmark, and pull request helps make `aarambh-voice-studio` stronger.

This project is ambitious: it combines speech, voice cloning, music generation, singing synthesis, mixing, safety, and Rust-native AI training. Please keep changes focused, tested, and aligned with the roadmap.

---

## Table of Contents

1. Code of Conduct
2. Ways to Contribute
3. Project Status
4. Setting Up the Workspace
5. Project Structure
6. Making a Change
7. Commit Messages
8. Testing Requirements
9. Documentation Requirements
10. Audio and Dataset Contributions
11. Safety-Sensitive Contributions
12. Working on Custom Kernels
13. Pull Request Process
14. Reporting Bugs
15. Suggesting Features
16. Crate Versioning
17. Questions

---

## Code of Conduct

This project follows a simple rule: be respectful. Constructive criticism of code and ideas is welcome; personal attacks are not.

By participating, you agree to follow `CODE_OF_CONDUCT.md`.

---

## Ways to Contribute

You do not need to write model code to contribute.

Useful contributions include:

- report a bug with a minimal reproduction
- improve documentation
- add examples
- write tests
- write benchmarks
- improve CLI help text
- improve error messages
- help with audio file handling
- add safe DSP utilities
- improve dataset validation
- review pull requests
- propose roadmap improvements through issues

For larger features, open an issue first.

---

## Project Status

`aarambh-voice-studio` is a roadmap-stage engineering project. The architecture and roadmap are intentionally detailed, but the implementation should proceed phase by phase.

Do not submit a PR that pretends an unfinished phase is complete. Mark work honestly as:

- planned
- scaffolded
- experimental
- implemented
- tested
- benchmarked
- release-ready

No pretrained checkpoints, voice packs, cloned voices, adapters, or generated assets should be committed to this repository.

---

## Setting Up the Workspace

### Prerequisites

- Rust stable, 1.80 or later
- Git
- `rust-analyzer` recommended
- Linux recommended for development
- Optional: CUDA/NVCC for later GPU work

### Clone and build

```bash
git clone https://github.com/AarambhDevHub/aarambh-voice-studio.git
cd aarambh-voice-studio

cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

### IDE setup

This is a standard Cargo workspace. Any IDE with `rust-analyzer` support should work. Open the root `aarambh-voice-studio/` folder.

---

## Project Structure

```text
aarambh-voice-studio/
├── Cargo.toml
├── crates/
│   ├── aarambh-voice-core/          # Configs, request/response types, errors
│   ├── aarambh-voice-codec/         # Neural audio codec and RVQ tokens
│   ├── aarambh-voice-data/          # Dataset loading and preprocessing
│   ├── aarambh-voice-nn/            # Transformer primitives and conditioning
│   ├── aarambh-voice-kernel/        # SIMD, CUDA prep, STFT kernels
│   ├── aarambh-voice-model/         # Voice, music, and singing models
│   ├── aarambh-voice-weights/       # SafeTensors and checkpoint I/O
│   ├── aarambh-voice-train/         # Training loops
│   ├── aarambh-voice-quant/         # Quantisation
│   ├── aarambh-voice-finetune/      # LoRA, QLoRA, DoRA
│   ├── aarambh-voice-speaker/       # Speaker encoder and voice design
│   ├── aarambh-voice-emotion/       # Emotion controls
│   ├── aarambh-voice-music/         # Music understanding and generation
│   ├── aarambh-voice-sing/          # Singing synthesis
│   ├── aarambh-voice-mix/           # Mixing and mastering
│   ├── aarambh-voice-compose/       # Lyrics-to-song composer
│   ├── aarambh-voice-safety/        # Consent, watermarking, guardrails
│   ├── aarambh-voice-eval/          # Evaluation harness
│   ├── aarambh-voice-control/       # Unified NaadRequest API
│   ├── aarambh-voice-inference/     # KV cache and streaming inference
│   └── aarambh-voice-serve/         # HTTP inference server
└── aarambh-voice-studio/            # CLI binary
```

Each crate should stay focused. If you are working on emotion embeddings, you should not need to modify the music generator in the same PR.

---

## Making a Change

### 1. Check for an existing issue

Search open issues before starting. If there is no issue for your change, open one first for anything larger than a typo or small documentation fix.

### 2. Fork and branch

```bash
git clone https://github.com/YOUR_USERNAME/aarambh-voice-studio.git
cd aarambh-voice-studio
git checkout -b fix/codec-token-shape
```

Branch naming conventions:

| Change type | Prefix | Example |
|---|---|---|
| Bug fix | `fix/` | `fix/codec-token-shape` |
| Feature | `feat/` | `feat/tts-tiny-forward` |
| Documentation | `docs/` | `docs/safety-policy` |
| Refactor | `refactor/` | `refactor/control-request-types` |
| Performance | `perf/` | `perf/stft-rustfft-buffer-reuse` |
| Tests | `test/` | `test/emotion-zero-intensity` |
| Build/CI | `ci/` | `ci/clippy-workspace` |

### 3. Make the smallest useful change

Do not mix unrelated changes. A PR that adds `NaadRequest` validation should not also redesign the music model.

### 4. Reference the roadmap

If your change maps to a roadmap phase, mention it in the PR description:

```text
Related to Phase 0: Workspace + core types
```

### 5. Format your code

```bash
cargo fmt
```

CI should reject unformatted code.

---

## Commit Messages

Use Conventional Commits:

```text
<type>(<scope>): <short description>
```

Examples:

```text
feat(core): add AudioDomain enum
fix(codec): validate token grid frame count
docs(readme): explain three-engine architecture
test(control): reject cloned voice without consent token
perf(kernel): reuse stft scratch buffers
```

Allowed types:

| Type | Use |
|---|---|
| `feat` | New feature or behavior |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `perf` | Performance improvement |
| `refactor` | Code restructuring without behavior change |
| `chore` | Build system or maintenance |
| `ci` | CI configuration |

Rules:

- Use imperative mood: `add`, `fix`, `update`.
- Keep the first line under 72 characters.
- Use the crate name as the scope without the `aarambh-voice-` prefix when possible.
- Reference issues in the footer: `Closes #42`.
- Breaking changes must include `BREAKING CHANGE:` in the footer.

---

## Testing Requirements

Every pull request that changes behavior must include tests.

### What to test

- new behavior
- bug reproductions
- config serialization
- shape invariants
- zero-length and boundary audio inputs
- invalid request validation
- deterministic output with fixed seeds
- safety gates for cloning and watermarking

### Where to put tests

- Small unit tests: `#[cfg(test)]` inside the relevant `src/*.rs` file.
- Integration tests: `tests/` at the crate root or workspace root.
- Audio fixture tests: use tiny synthetic clips, not copyrighted or personal audio.

### Running tests

```bash
cargo test --workspace
cargo test -p aarambh-voice-core
cargo test -p aarambh-voice-control reject_cloning_without_consent
```

### Clippy

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Any clippy warning should be treated as a CI failure.

---

## Documentation Requirements

Public APIs need documentation.

- Every `pub` item should have a `///` doc comment.
- Non-trivial APIs should include examples.
- Request validation rules must be documented.
- Unsafe code must explain invariants clearly.
- Audio assumptions must be explicit: sample rate, channels, frame size, dtype, normalization.

Check docs locally:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

---

## Audio and Dataset Contributions

Do not commit large audio files, copyrighted songs, personal voice recordings, voice packs, or cloned speaker samples.

Allowed examples:

- tiny generated sine waves
- synthetic test WAV files
- public-domain mini fixtures with clear license notes
- JSONL manifest examples with fake paths
- small text-only configs

Dataset loaders must:

- validate file existence
- validate sample rate
- handle stereo/mono conversion explicitly
- reject corrupted audio gracefully
- avoid panics on malformed metadata
- record license/source fields where relevant

---

## Safety-Sensitive Contributions

Voice cloning, singing-voice cloning, watermarking, guardrails, and consent handling are safety-sensitive areas.

For these areas:

- open an issue before implementation
- keep PRs small and reviewable
- include tests for allowed and rejected paths
- never add bypass flags for consent checks
- never log raw reference audio
- never add example cloned voices without explicit permission
- never weaken watermarking silently
- document any safety tradeoff clearly

Examples of required tests:

```text
VoiceSpec::Cloned without consent_token => rejected
Watermarked audio => detector returns true
Unwatermarked audio => detector returns false
Invalid lyrics/content => clear validation error
```

---

## Working on Custom Kernels

`aarambh-voice-kernel` may contain CPU SIMD, CUDA build prep, and low-level DSP code.

Rules:

- Keep unsafe code isolated.
- Provide scalar fallback paths.
- Runtime feature detection must be tested.
- Tests must pass without CUDA.
- CUDA/NVCC should be optional.
- Audio numerical differences must be bounded with explicit tolerances.

Example commands:

```bash
cargo test -p aarambh-voice-kernel
cargo bench -p aarambh-voice-kernel
```

---

## Pull Request Process

### Before opening

Run:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

All must pass.

### Opening the PR

- Use a Conventional Commits title.
- Link the related issue.
- Reference the roadmap phase.
- Explain what changed and why.
- Include screenshots/logs only when useful.
- For audio output, describe the test fixture rather than uploading large audio.
- Mark incomplete work as Draft.

### Review process

- At least one maintainer approval is required.
- Address review comments directly.
- Rebase on `main`; avoid merge commits in feature branches.
- The maintainer may squash-merge.

---

## Reporting Bugs

Open an issue using the bug report template. Include:

1. What you expected.
2. What happened.
3. Minimal reproduction.
4. Full error output.
5. Rust version: `rustc --version`.
6. OS and CPU/GPU details.
7. Active feature flags.
8. The exact crate or CLI command involved.
9. For audio bugs: sample rate, channel count, duration, and input format.

Do not attach private voice recordings or copyrighted music to public issues.

---

## Suggesting Features

Open a feature request issue. Include:

1. The use case.
2. Why current architecture does not solve it.
3. Proposed API or CLI command.
4. Safety implications.
5. Alternatives considered.
6. Related roadmap phase.

Feature requests that only say “add this model” without explaining the use case may be closed or asked for more detail.

---

## Crate Versioning

`aarambh-voice-studio` should follow Semantic Versioning once releases begin.

- Patch (`1.0.x`) — bug fixes only.
- Minor (`1.x.0`) — backward-compatible features.
- Major (`x.0.0`) — breaking changes.

During early roadmap phases, tags may use phase milestones such as:

```text
v0.1.0-phase0
v0.1.0-phase1
v0.1.0-phase5
```

All sub-crates should share the same workspace version.

---

## Questions

If you are unsure whether a bug, feature, model idea, dataset, or safety change fits the project, open an issue and ask before building it.

