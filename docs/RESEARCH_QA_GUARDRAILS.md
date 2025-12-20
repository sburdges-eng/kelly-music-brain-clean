# Research AI QA, Guardrails, Status

## Scope
- Applies to all research agents (Kelly emotion, Lyrics/LLM, Chord/Groove/Section).

## Sources & Access
- Allowlisted only: vetted docs, project MDs/JSONs, selected Git/GitHub/papers; no broad scraping.
- Local cache default; cloud retrieval only with user permission.

## Behavior
- Citations required; return “don’t know” if unsupported.
- Guardrails: no medical/diagnostic advice; avoid copyrighted/unsafe content; refuse unlicensed sources.
- Status/pause panel: shows sources used, last update, citations, pause/resume control.
- Personas: Kelly voice for writer-block; Dee voice for co-producer/timing contexts.

## Tests/Checks
- Hallucination and citation enforcement.
- Mapping validation: emotion → music hints (tempo/mode/dynamics), chord-emotion function, groove bounds.
- Latency/throughput sanity for retrieval/indexing.
- Safety filters: harmful content, unlicensed/copyrighted material.

## Outputs
- Markdown/KB notes: design guides, checklists, mappings, with citations; no direct user-facing generation.

## Citations (format)
- Inline short refs: `[1][theory]`, `[2][practice]`, `[3][anecdotal]`, `[4][tooling]`.
- End-of-section bibliography: numbered list with source type tags.

## Not allowed
- No medical/diagnostic advice; no unlicensed/non-allowlisted sources; no uncited claims; no broad scraping; no user-facing generation from research agents.
