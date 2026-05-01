# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `src/`.
- CLI entrypoint: `src/main.py` (pipeline orchestration and argument parsing).
- Pipeline modules: `transcription.py`, `translation.py`, `timing_review.py`, `subtitles.py`, `video_processing.py`.
- Shared types/utilities: `models.py`, `utils.py`, `retry.py`, `gemini_sdk.py`.
- Tests live in `tests/` (notably `test_timing_review.py` and `test_transcription_multimodal.py`).
- Runtime folders: `input/` for source media, `output/` for generated artifacts, and `references/` for glossary/context files.
- Local design notes and implementation plans may live under `docs/`, but `docs/` is ignored and not tracked in Git. Put durable setup, workflow, or operational guidance in `README.md` or `AGENTS.md`.

## Build, Test, and Development Commands
- Install dependencies: `pip install -r requirements.txt`
- Install test dependency (dev): `pip install pytest`
- Run tests from repo root: `PYTHONPATH=src python -m pytest tests/ -v`
- Run full pipeline: `python src/main.py <video_file>.mp4`
- Helpful check: `python src/main.py --help`

`ffmpeg` must be installed and available on PATH.

## Coding Style & Naming Conventions
- Language: Python 3.9+.
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
- Add or update focused unit tests for any logic changes, especially around transcription normalization, timing repair/review, and translation chunking/fallback behavior.

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

## Commit & Pull Request Guidelines
- Prefer concise, imperative commit subjects (e.g., `Add stage cooldown between API-heavy steps`).
- Conventional prefixes are used in history and recommended when useful: `feat:`, `refactor:`, `docs:`.
- PRs should include:
  - what changed and why
  - key CLI flags/behavior impacted
  - test evidence (command + result)
  - sample output path or screenshots only when UI/output formatting changes are relevant

## Security & Configuration Tips
- Keep secrets in `.env` only (`GEMINI_API_KEY`, `OPENROUTER_API_KEY`); never commit keys.
- Avoid committing large generated artifacts under `output/` unless explicitly required for review.
