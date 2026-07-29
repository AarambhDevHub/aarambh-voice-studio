# Changelog

## [0.1.0] - 2026-07-29

### Added

- Phase 0 — Workspace + Core Types.
- 24-crate Cargo workspace with shared dependency metadata and release profile.
- `aarambh-voice-core` crate with:
  - `AudioDomain` enum (Speech, Music, Singing).
  - `ModelConfig` with four preset scales (Tiny, Small, Medium, Large) per ARCHITECTURE.md §5.
  - `NaadRequest`, `VoiceSpec`, `EmotionSpec`, `SingingSpec`, `MusicSpec`, `MixSpec`, `AudioOutputFormat`, `ConsentSpec` stub types.
  - `AarambhVoiceError` via thiserror (one variant per crate).
  - Full doc-comment coverage on every public item.
- 23 scaffold crates, each re-exporting `aarambh_voice_core`.
- `aarambh-voice-studio` binary crate.
- `.github/` directory:
  - `FUNDING.yml`, `PULL_REQUEST_TEMPLATE.md`.
  - `ISSUE_TEMPLATE/bug_report.md`, `ISSUE_TEMPLATE/feature_request.md`.
  - `workflows/ci.yml` (MSRV check, quality gates, RustSec audit).
  - `workflows/release.yml` (tag-triggered v0.1.0 source release).
  - `release-notes/v0.1.0.md`.
- `CHANGELOG.md`.
- `.gitignore`.
- `docs/README.md`, `docs/VOICE_STUDIO_AUDIO_ML_TERMINOLOGY_AND_DATASET_GUIDE.md`, `docs/VOICE_STUDIO_MATH_FORMULAS_GUIDE_PART1.md`, `docs/VOICE_STUDIO_MATH_FORMULAS_GUIDE_PART2.md`.

### Quality

- `cargo check --workspace --all-targets --locked` — zero errors, zero warnings.
- `cargo clippy --workspace --all-targets --locked -- -D warnings -D clippy::undocumented_unsafe_blocks` — clean.
- `cargo test --workspace --no-fail-fast --locked` — 8 tests pass.
- `RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc --workspace --no-deps --locked` — clean.

### Guarantees

- All scaffold crates compile on Rust 1.89 (MSRV).
- No pretrained checkpoints, model artifacts, or generated data are committed.
- Every public API item is documented.
