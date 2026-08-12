# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `src/`.
- CLI entrypoint: `src/main.py` is a compatibility wrapper; packaged CLI code lives in `src/hermecho/cli.py`.
- Pipeline orchestration lives in `src/hermecho/pipeline.py`.
- Pipeline modules: `transcription.py`, `translation.py`, `subtitles.py`, and `video_processing.py`.
- Translation uses OpenRouter through the OpenAI-compatible `openai` package. The former Gemini SDK helper was removed; do not add new runtime dependencies on `gemini_sdk.py`.
- Shared helpers live in `prompts.py`, `progress.py`, and `utils.py`.
- Tests live in `tests/` (notably `test_main_cli.py`, `test_transcription.py`, `test_translation.py`, and `test_video_processing.py`).
- Runtime folders: `input/` for source media, `output/` for generated artifacts, and `references/` for glossary/context files.
- Local design notes and implementation plans may live under `docs/`, but `docs/` is ignored and not tracked in Git. Put durable setup, workflow, or operational guidance in `README.md` or `AGENTS.md`.

## Build, Test, and Development Commands
- Install dependencies: `python -m pip install -e ".[dev]"`
- Compatibility install: `python -m pip install -r requirements.txt`
- Update the project Conda environment: `conda run -n hermecho python -m pip install -e ".[dev]"`
- Run tests from repo root: `conda run -n hermecho python -m pytest tests/ -q`
- Run full pipeline: `conda run -n hermecho hermecho <video_file>.mp4`
- Helpful check: `conda run -n hermecho hermecho --help`

`ffmpeg` must be installed and available on PATH.
Normal transcription auto-detects the source language from the first 30 seconds by default; pass `--language` to force a specific language.
On macOS, use the default `Heiti TC` font for burn-in; `PingFang TC` may render Chinese glyphs as replacement boxes through FFmpeg/libass.

## Coding Style & Naming Conventions
- Language: Python 3.11+.
- Use 4-space indentation, type hints, and clear function names.
- Follow existing naming patterns:
  - modules/files: `snake_case.py`
  - functions/variables: `snake_case`
  - classes: `PascalCase`
  - constants: `UPPER_SNAKE_CASE`
- Keep pipeline stages modular; prefer adding behavior to the stage module that owns it instead of expanding `main.py`.
- No formatter/linter config is committed; match surrounding style and keep imports/typing consistent with nearby code.

## Testing Guidelines
- Framework: `pytest` (tests are written in `unittest.TestCase` style and executed by pytest).
- Test files: `tests/test_*.py`; test methods: `test_*`.
- Add or update focused unit tests for any logic changes, especially around transcription normalization, OpenRouter translation chunking/fallback behavior, and subtitle burn-in handling.

## Session Workflow
- At the start of each agent session, check the current branch and worktree state before editing.
- Review the relevant user request, plan, docs, or prior context so the session begins from the current intended direction.
- Create a new branch only when the session is for a new feature or bug fix; use a concise descriptive branch name.
- Before ending a session, review changed files and `git diff` so the final state is understood.
- Run relevant tests for the work completed, or document why tests were not run.
- Commit completed session changes with a concise imperative message.
- Before pushing or merging a branch, inspect the branch-only changes against the base branch.
- Update `README.md` before push or merge when changes affect user-facing behavior, setup, commands, dependencies, examples, documented workflow, or user-facing output.
- Leave `README.md` unchanged only after confirming the branch has no documentation-impacting changes.
- For a Hermecho Cloud portrait rollout, install the compatible Hermecho release on the processor Mac before deploying Cloud; the pipeline owns the portrait subtitle cue limit.

## Commit & Pull Request Guidelines
- Prefer concise, imperative commit subjects (e.g., `Add stage cooldown between API-heavy steps`).
- Conventional prefixes are used in history and recommended when useful: `feat:`, `refactor:`, `docs:`.
- PRs should include:
  - what changed and why
  - key CLI flags/behavior impacted
  - test evidence (command + result)
  - sample output path or screenshots only when UI/output formatting changes are relevant

## Security & Configuration Tips
- Keep secrets in `.env` only (`OPENROUTER_API_KEY`); never commit keys.
- Avoid committing large generated artifacts under `output/` unless explicitly required for review.

## Agent skills

### Issue tracker

Issues and specs are tracked in GitHub Issues for `coletangsy/hermecho`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the canonical triage roles `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository with a root `CONTEXT.md` and system-wide ADRs under `docs/adr/`. See `docs/agents/domain.md`.
