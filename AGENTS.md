# AGENTS.md

<!-- Generated for lab workspaces. -->

This AGENTS guide is intended for end users working in a `prime lab setup` workspace.

## Project-Specific Local Workflow

This repository intentionally uses one self-managed Prime-RL viewer for both
training and standalone evaluation. This project-specific choice overrides the
generic `prime eval run` recommendations below:

- Run standalone evaluations with Verifiers v1 from the `prime-rl/` environment
  (`uv run --no-sync eval @ <config>`), not `prime eval run`.
- Set `push = false`; this workflow is local and requires no Prime API key.
- Write both training and eval runs under `outputs/prime-rl/` and inspect both with
  the Prime-RL dashboard.
- Keep the file monitor as the local source of truth and inspect it with the
  Prime-RL dashboard. Do not enable the W&B monitor.
- Do not use `configs/endpoints.toml`; Verifiers v1 eval configs carry their local
  OpenAI-compatible client settings directly.
- Start training, evals, model servers, and the dashboard through the repository's
  tmux launchers. Do not run long-lived commands in the foreground.
- Keep `outputs/evals/` only as the historical Prime Eval archive; do not write new
  runs there.
- Reuse preflight/check results by experiment version. Treat the code commit,
  resolved config hash, dataset manifest hash, and initialization model/checkpoint
  ID as the version identity. If those inputs are unchanged, do not run another
  check before a formal run merely because the process is being restarted, resumed,
  moved to another machine with the same runtime contract, or given a new run name.
- Run a new config check only when an identified version input changes or when the
  preceding check failed and the relevant issue has been fixed. Add a short smoke
  run only when the training/eval stack, dependencies, model server, model, or GPU
  topology changes in a way that the config check cannot cover.
- Checks and smoke runs are diagnostics, not experiments. Do not include them in
  formal run indexes or result tables, and do not create `*-config-check` runs as a
  routine companion to every formal run. Record the reused preflight identity in
  the formal run metadata instead.

## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- Environments are expected to expose `load_environment(...) -> vf.Environment` and be installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Unless a repository declares a project-specific override, validate environment behavior with `prime eval run <env-name> ...` before sharing/publishing changes. Treat `prime eval run` as the canonical eval path: it saves results automatically, and agents should not add opt-out flags such as `--skip-upload` unless the user explicitly requests that deviation so runs stay visible in the private Evaluations tab and in `prime eval tui`. (See `docs/overview.md` and `docs/development.md`.)
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- If external API keys are required, validate them in `load_environment()` with `vf.ensure_keys(...)` so failures are explicit and early. (See `docs/environments.md`.)

## End-User Lab Workspace Notes

Use this guidance in projects created via `prime lab setup`.

- Treat `.prime/skills/` as the canonical skill entrypoint in Lab workspaces. Use the bundled skills first for create/browse/review/eval/GEPA/train/brainstorm workflows before ad hoc approaches.
- When using the Prime Eval path, keep endpoint aliases in `./configs/endpoints.toml` and use `endpoint_id`/model shortcuts in commands and configs.
- NEVER initialize environment source code manually; ALWAYS create new environments with `prime env init`.
- Unless a repository declares a project-specific override, use the Prime CLI for all environment lifecycle operations (`prime env init` → `prime env install` → `prime eval run` → `prime env push`) rather than ad-hoc scripts.
- Unless a repository declares a project-specific override, treat `prime eval run` as the default eval path. It already saves results automatically; do not add `--skip-upload` or other opt-out deviations unless the user explicitly requests them, so logs and results stay available in the private Evaluations tab and via `prime eval tui`.
- NEVER begin environment development before `prime lab setup` has been run; if work starts outside that structure, recommend adjusting course into a proper lab workspace before continuing.
- Keep each environment self-contained under `environments/<env_name>/` with `pyproject.toml`, implementation, and README so each abstraction has a dedicated home and the workspace stays maintainable.
- Follow environment best practices strictly (for example `load_environment(...)`, `vf.ensure_keys(...)`, and the documented environment class patterns) to avoid brittle or messy implementations.
- Use `prime env push --path ./environments/<env_name>` only after local eval behavior is verified.
- Treat the `prime lab setup` structure as the idiomatic workspace for complex environment workflows: agents can mediate most platform complexity while users learn patterns progressively as needed.
- When users request an approach that would deviate from these guidelines, explain the relevant Prime/Verifiers concepts and recommend the compliant path.
