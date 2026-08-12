---
name: run-hermecho-job
description: Run, resume, monitor, diagnose, and verify long-running Hermecho video transcription, Traditional Chinese translation, subtitle delivery, ASR comparison, or sentence-first comparison jobs. Use for Hermecho CLI operations, especially interrupted runs, checkpoint reuse, Delivery Gate verification, and comparison evidence generation. Do not use for changing Hermecho source code or approving comparison evidence.
---

# Run Hermecho Job

Operate the repository's existing CLI safely. Do not add another runner, duplicate pipeline logic, or commit generated media.

## Select the mode

Choose exactly one mode from the request:

- Standard pipeline: `conda run -n hermecho hermecho <video_filename>` with only the requested flags.
- ASR Comparison Run: `conda run -n hermecho python -m hermecho.asr_comparison <video_path>`.
- Sentence-first Comparison Run: `conda run -n hermecho python -m hermecho.sentence_first_comparison <video_path>`.

Use `README.md` and each command's `--help` as the current source of truth for flags and artifacts. For a standard input outside `input/`, pass its basename as `video_filename` and its parent with `--input_dir`; do not copy the media merely to satisfy the CLI.

## Preflight

1. Work from the Hermecho repository root. Read `AGENTS.md`, `README.md`, and the relevant request or plan; run `git status --short --branch` without altering tracked files.
2. Resolve the exact input file, mode, requested flags, output directory, and expected artifacts. Require the input to exist as a readable regular file. Ask only when one of these materially changes the job.
3. Run the selected command's `--help` inside the `hermecho` Conda environment. If the environment, package, or selected module is unavailable, report the exact failure and stop; do not install or upgrade dependencies without approval.
4. Before an ASR Comparison Run or any standard or sentence-first command that explicitly selects `--transcription-backend mlx`, require Apple Silicon and use Hermecho's `validate_mlx_backend` check for the selected model. Report its actionable error and stop when MLX cannot run.
5. Confirm `ffmpeg` is available. For burn-in and Comparison Runs, also confirm the `subtitles` filter is present.
6. For translated runs, confirm `OPENROUTER_API_KEY` is available without printing its value, and confirm the Locked Terms file exists and is a valid JSON object. Keep the reference file separate from Locked Terms.
7. Confirm the output parent is writable and has reasonable free space. Require an empty output directory for a Comparison Run; never delete or overwrite an existing comparison directory without explicit permission.

## Run and monitor

1. Show the resolved command before starting it. Preserve the exact command for retry or handoff.
2. Start one process in a persistent execution session and monitor that session. Never launch a duplicate because output is temporarily quiet.
3. Report meaningful stage transitions and keep the user updated during long stages. Avoid busy polling.
4. On interruption or retry of a standard pipeline, run the same command. Let Hermecho reuse its matching transcription and Translation-Gate-approved chunk checkpoints.
5. Treat Comparison Runs as non-resumable. Preserve a failed run's non-empty output for diagnosis; after approval, rerun the full comparison with a new empty `--output-dir` instead of deleting or reusing partial evidence.
6. Do not add `--force` unless the user explicitly requests a full recomputation or evidence shows checkpoint reuse itself is the problem. Explain the lost work before using it.
7. On failure, record the exit status, last completed stage, exact error, checkpoint path when applicable, and artifacts already written. Diagnose that stage before retrying; do not restart blindly.

## Verify a standard run

Compare the output inventory from before and after the run; do not select an artifact only because it is the newest unrelated file.

- Transcribe-only: require a non-empty `*_transcript.srt`.
- SRT-only: require a non-empty `*_subtitles.srt` and matching `*_delivery_gate.txt`.
- Full pipeline: require those translated artifacts plus a non-empty `*_translated.mp4` that `ffprobe` can read.
- Translated runs: read the Delivery Gate report. Treat Structural Defects or a blocked Translation Gate as failure; report Warnings and unresolved Repair Limits even when Best-effort Delivery produced files.
- Confirm the expected `.hermecho-checkpoint.json` exists when the run created reusable progress.

## Verify a Comparison Run

1. Require `manifest.json`, `comparison.json`, `review.md`, and every artifact named by the report or manifest.
2. Confirm the comparison changed only its declared variable and that the expected processes completed.
3. Confirm the Review Composite is non-empty and readable with `ffprobe`, and inspect any transcript diff and Delivery Gate reports named by the evidence.
4. Report the generated review checklist as pending. Never fill in reviewer identity, mark Human Approval, or promote a candidate on the user's behalf.

## Finish

Report the command, mode, elapsed result when available, reused checkpoints, Delivery Gate status, artifact paths, and anything still requiring human review. Leave generated output uncommitted. Leave credentials, source media, and tracked repository files unchanged.
