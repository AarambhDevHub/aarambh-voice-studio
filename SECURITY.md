# Security Policy

`aarambh-voice-studio` is a safety-sensitive AI audio project. It includes planned support for text-to-speech, voice cloning, emotional speech, singing synthesis, music generation, and full-song composition. Security reports are taken seriously.

---

## Supported Versions

| Version | Supported |
|---|---|
| `main` | ✅ Development branch |
| `v0.x` phase tags | ✅ Best-effort security fixes |
| `v1.0.x` | ✅ Supported after production release |

During roadmap development, the active `main` branch and the latest phase tag are the primary supported targets.

---

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

1. **Do not open a public GitHub issue with exploit details.**
2. Use GitHub private vulnerability reporting or open a private GitHub Security Advisory for this repository.
3. If private reporting is unavailable, open a minimal public issue asking for maintainer security contact, without exploit details.
4. Include a clear description of the vulnerability.
5. Include steps to reproduce if possible.
6. Include affected commit, tag, crate, feature flags, and OS.
7. For audio-related vulnerabilities, include metadata such as sample rate, format, duration, and whether the audio can be shared safely.

We aim to acknowledge receipt within 48 hours and provide an initial fix plan within 7 days.

---

## Security Scope

Security concerns for `aarambh-voice-studio` include, but are not limited to:

- checkpoint or model weight tampering
- unsafe checkpoint deserialization
- malicious SafeTensors or metadata handling
- malformed audio file parsing
- denial of service through oversized or corrupted audio
- denial of service through extremely long prompts, lyrics, melodies, or duration requests
- training data poisoning
- prompt/control-layer injection through `NaadRequest`
- bypassing consent checks for voice cloning
- bypassing watermarking or watermark detection
- leaking raw reference audio through logs, telemetry, caches, or crash dumps
- unauthorized voice cloning workflows
- unsafe handling of user-uploaded reference voices
- unsafe handling of generated singing voices
- HTTP server request smuggling, path traversal, or streaming abuse
- dependency vulnerabilities in audio decoding, serving, or tensor code
- unsafe Rust or custom-kernel memory safety bugs

---

## AI Audio Safety Scope

Because this project can generate voice and singing, safety issues also include misuse-enabling behavior.

Please report privately if you find a way to:

- clone a voice without a required consent token
- remove or bypass generated-audio watermarking
- make the system output unwatermarked audio through an alternate path
- cause audit logs to store raw reference audio
- impersonate a real person through a built-in example or fixture
- bypass content guardrails for speech or lyrics
- abuse the system for hidden identity deception

Do not publish working bypasses publicly before maintainers have had time to fix them.

---

## Out of Scope

The following are usually not treated as security vulnerabilities unless they create a concrete exploit path:

- low-quality generated audio
- model hallucination or wrong musical tags
- poor singing pitch accuracy
- non-sensitive crashes in unfinished roadmap phases
- missing features marked as planned
- benchmark disagreements
- normal model quality limitations

If you are unsure, report privately or open a minimal public issue without exploit details.

---

## Responsible Disclosure

We follow a 90-day responsible disclosure policy. Please give maintainers reasonable time to investigate and fix the issue before public disclosure.

For high-risk issues involving unauthorized voice cloning, watermark bypass, private audio leakage, or remote code execution, please avoid sharing proof-of-concept exploit code publicly until a fix is available.

---

## Security Design Principles

`aarambh-voice-studio` aims to follow these principles:

- no pretrained cloned voice packs in the repository
- no private voice data committed to source control
- consent required for non-preset cloning
- synthetic audio watermarking by default
- audit logs store hashes, not raw audio
- safe defaults for CPU-only builds
- no mandatory network access for local inference
- corrupted audio should return errors, not panics
- unsafe code must be isolated and justified
- external dependencies should be pinned and reviewed

---

## Maintainer Notes

Before a public release, maintainers should verify:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo audit
```

Security-sensitive crates should receive extra review:

- `aarambh-voice-safety`
- `aarambh-voice-control`
- `aarambh-voice-codec`
- `aarambh-voice-data`
- `aarambh-voice-kernel`
- `aarambh-voice-serve`
- `aarambh-voice-weights`

