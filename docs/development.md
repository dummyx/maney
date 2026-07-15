# Development Guide

This guide is for contributors extending the local research pipeline. Repository safety and research
invariants are defined in [`AGENTS.md`](../AGENTS.md); human behavior and contracts live in this
documentation set.

## Environment

The baseline targets Python 3.12 and Apple Silicon but remains CPU-capable.

```bash
uv sync --locked
```

Optional transformer dependencies:

```bash
uv sync --extra nlp
```

Do not make the baseline depend on PyTorch or a model download.

## Repository map

```text
configs/                    bundled typed YAML examples
data/samples/               tiny redistributable fixtures
docs/                       human documentation
src/nlp_trader/
  backtest/                 replay, costs, and metrics
  data/                     local decoding, Parquet, raw store, model store, fixtures
  features/                 traditional/text feature and label builders
  models/                   walk-forward baselines and evaluation
  nlp/                      preprocessing, linking, sentiment, optional transformer/LLM,
                            semantic verifier, and DecisionRound ledger
  paper/                    programmatic in-memory simulator
  portfolio/                eligibility, construction, constraints, and exposures
  calendars.py              bounded exchange-calendar behavior
  cli.py                    Typer commands and runtime overrides
  config.py                 strict immutable configuration
  pipeline.py               dependency graph and end-to-end orchestration
  providers.py              protocols and local provider implementations
  research.py               run context, hashing, and manifests
  reports.py                Markdown research notes
  schemas/                  typed boundary records
tests/
  unit/                     component contracts and edge cases
  integration/              generated-data stage composition
  property/                 generated invariant checks across broad input ranges
  regression/               reproducibility and leakage sentinels
```

Production logic belongs under `src/`. Notebooks may call production modules, but production modules
must not depend on notebooks.

## Quality gates

Run before finalizing a substantive change:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
uv run nlp-trader smoke --config configs/sample.yaml
```

The [GitHub Actions quality workflow](../.github/workflows/quality.yml) applies these gates on every
push and pull request. Ubuntu runs formatting, lint, strict type checks, the full test suite, and the
sample smoke. A macOS arm64 runner repeats the baseline tests and sample. A Windows runner executes
the broker boundary tests and validates the bundled non-secret broker config. All three jobs sync
the exact lockfile before switching uv to offline, no-sync execution, which separates dependency
installation from the no-dependency-access checks. This is not an operating-system-level network
sandbox.

The macOS job verifies the baseline on Apple Silicon. It deliberately omits the optional `nlp`
extra, so it does not claim that PyTorch or MPS inference was exercised.

Tests must not require vendor credentials, paid APIs, network access, CUDA, or MPS. Use fixed seeds
and tolerant floating-point assertions. Hypothesis property tests cover cost monotonicity,
post-construction portfolio limits, determinism, and point-in-time provenance rejection.

If the full suite is impractical during a narrow edit, run targeted tests while working, then run the
full gates before handoff. If a gate truly cannot run, state exactly what was skipped and why.

## Code conventions

- Type public functions and keep errors actionable.
- Keep I/O at boundaries and transformations pure where practical.
- Avoid global mutable state and implicit network calls in imports/constructors.
- Use standard logging in library code; CLI output is deliberate.
- Put runtime choices in typed config rather than hardcoding paths, symbols, hosts, model IDs, or
  strategy parameters.
- Preserve user changes in a dirty tree and avoid unrelated rewrites.
- Use Polars lazy scans for large Parquet sources; filter/project before joins.
- Keep batch sizes and local-development bounds configurable.
- Centralize optional MPS/CPU selection in `utils/device.py`.

## Add or change an input provider

1. Define or reuse a provider protocol in `providers.py`.
2. Implement a local fixture adapter first.
3. Validate into typed schemas at the boundary.
4. Preserve exact source bytes before parsing.
5. Define `available_at` and revision semantics explicitly.
6. Add unit contract tests and generated-data integration coverage.
7. Document fields in [Input data](input_data.md) and timing in [Data contracts](data_contracts.md).
8. Keep any external call behind an explicit CLI action with licensing, retry, rate-limit, and cache
   behavior. Current pipeline stages are local-only.

Do not place credentials in config snapshots or manifests. The current local providers do not
consume the reserved environment-secret settings.

## Add a feature

1. State the economic/research rationale and exact window.
2. Identify every input and its historical availability.
3. Add a missingness field when zero and unavailable have different meanings.
4. Ensure `available_at <= asof_ts` for every contributing input.
5. Do not calculate a forward label inside feature code.
6. Add boundary, warm-up, and leakage-sentinel tests.
7. Decide deliberately whether the baseline model’s prefix discovery should include it.
8. Version the feature set and update [Features and models](features_and_models.md).

## Add or change a label

1. Start strictly after the decision.
2. Use exact exchange sessions, not weekdays.
3. Keep raw simulated fill prices separate from causal return adjustment.
4. Record outcome availability and missing required sessions.
5. Add off-by-one, holiday, terminal-window, and partial-cross-section tests.
6. Update [Data contracts](data_contracts.md) and [Research protocol](research_protocol.md).

## Change a model

- Retain traditional-only, text-only, combined, and naive comparison paths.
- In LLM augment mode, also retain the complete canonical `llm`, `traditional_llm`, and `all`
  ablation paths without changing the conventional family meanings.
- Keep walk-forward label availability strict.
- Apply embargo/purging where overlapping outcomes require it.
- Record model version, selected features, parameters, and training provenance.
- Preserve a no-network deterministic baseline.
- Put transformer/deep-learning packages behind an optional extra.

Add tests showing that future labels cannot change earlier fitted state or predictions.

## Change portfolio or backtest logic

- Keep signal, portfolio, execution, and reporting layers separate.
- Enforce constraints after any asymmetric clipping or resizing operation.
- Model both entry and exit costs and turnover.
- Use only decision-known liquidity for ex-ante sizing.
- Fail rather than select assets based on future outcome coverage.
- Retain trades, positions, costs, rejects, exposures, and assumptions in replay output.
- Update [Backtesting](backtesting.md) in the same change.

No backtest or paper change authorizes live routing.

## Add optional transformer behavior

- Keep `torch` and `transformers` optional.
- Batch inference and use centralized MPS/CPU selection.
- Default to local model files and no test downloads.
- Cache by canonical text/model/config identity.
- Validate score, label, and confidence outputs.
- Add injected-predictor/cache tests and a small golden fixture.
- Version the model and decoding behavior in the run snapshot.

## Change optional generative LLM behavior

- Keep the baseline independent of PyTorch and keep model loading local-files-only with remote custom
  code disabled.
- Preserve current-source-only prompts unless a separately designed, point-in-time evidence store is
  added; do not imply that RAG, tools, or routing exist.
- Keep conventional sentiment/event fields separate from `llm_*` fields. Augmentation must retain the
  six canonical learned-family ablation.
- Version and hash model, prompt, schema, verifier, decoding, and cache identity.
- Persist each newly generated attempt before parsing, copy every consumed successful/cache response
  into the run, and keep licensed/private text out of git.
- Treat raw confidence as uncalibrated feature data, never a probability, signal magnitude, position
  size, or portfolio weight.
- Keep verifier claims precise: deterministic identity, timing, horizon, evidence-reference, and
  numeric-token checks do not establish semantic truth.
- Write DecisionRounds exclusively and replay-verify canonical JSON, timestamps, unique content IDs,
  and hashes. Do not populate tool, calibration, portfolio, risk, order, or outcome fields until an
  explicit later design extends that boundary.
- Add injected-generator/cache/failure-before-parse, leakage, verifier, feature-family, ablation, and
  DecisionRound tamper/replay tests without requiring a model download or MPS.

## Documentation ownership

| Change | Update |
|---|---|
| Setup or top-level capability | `README.md`, `docs/getting_started.md` |
| Config field or validation | `docs/configuration.md` and bundled configs |
| Input/derived schema or timestamp | `docs/input_data.md`, `docs/data_contracts.md` |
| CLI stage or dependency | `docs/workflows.md` |
| Artifact or metric | `docs/outputs.md` |
| Component/data flow | `docs/architecture.md` |
| Feature/model behavior | `docs/features_and_models.md`, `docs/research_protocol.md` |
| Fill, cost, constraint, or metric semantics | `docs/backtesting.md` |
| Licensing, privacy, secrets, or trading boundary | `docs/compliance.md` |
| Durable agent convention | `AGENTS.md` or closest nested `AGENTS.md` |

Avoid copying the same detailed reference into several files. Link to the canonical owner and keep a
short warning where visibility matters.

## Definition of done

- Requested behavior works without an unnecessary rewrite.
- Point-in-time, raw-data, licensing, and research-only invariants remain intact.
- Behavior changes have focused tests and updated human documentation.
- Quality gates and the smallest relevant smoke path pass.
- No secret, raw paid data, model artifact, report, or cache is added to git.
- The final handoff names changes, commands, checks, and remaining limitations without overstating
  performance.

Return to the [documentation home](README.md).
