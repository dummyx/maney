# Documentation

You do not need to read every page before running NLP Trader. Start with the path that matches the
decision in front of you; the exact schemas and invariants remain available when you need them.

For a plain-language introduction in Japanese, read the [Japanese README](../README.ja.md).

## Pick a reading path

### I want to run it

1. [Getting started](getting_started.md)
2. [Configuration reference](configuration.md)
3. [Input data guide](input_data.md)
4. [Workflows and commands](workflows.md)
5. [Outputs and artifacts](outputs.md)
6. [Troubleshooting](troubleshooting.md)

For a permitted local Japanese cash-equity export, follow the [Japan baseline](japan_baseline.md)
after the synthetic sample works.

### I want to evaluate a strategy

1. [Concepts and glossary](glossary.md)
2. [Data contracts](data_contracts.md)
3. [Features and models](features_and_models.md)
4. [Research protocol](research_protocol.md)
5. [Backtesting assumptions](backtesting.md)
6. [Compliance and safety](compliance.md)

For the optional governed local proposal workflow, read
[Workflows](workflows.md#local-research-agent-study), then
[Research protocol](research_protocol.md#local-agent-study-protocol).

### I want to change the code

1. [Architecture](architecture.md)
2. [Development guide](development.md)
3. [Data contracts](data_contracts.md)
4. [Research protocol](research_protocol.md)
5. [`AGENTS.md`](../AGENTS.md) for repository guardrails

### I want to operate the standalone broker adapter

1. [Broker integration](broker.md)
2. [Compliance and safety](compliance.md)
3. [Troubleshooting](troubleshooting.md)

## Document index

| Document | What it answers |
|---|---|
| [Getting started](getting_started.md) | How do I install, run, and inspect the synthetic sample? |
| [Japan baseline](japan_baseline.md) | How do I normalize a permitted J-Quants V2 or equivalent export into the strict XJPX contract? |
| [Concepts and glossary](glossary.md) | What do bronze, `available_at`, decision time, and horizon mean? |
| [Configuration](configuration.md) | What does each YAML section control? |
| [Input data](input_data.md) | Which files and columns are accepted? |
| [Workflows](workflows.md) | Which command should I run, and what prerequisites run with it? |
| [Outputs](outputs.md) | Where are reports, manifests, models, and replay logs? |
| [Architecture](architecture.md) | How do components and storage layers fit together? |
| [Data contracts](data_contracts.md) | What are the exact schema and point-in-time invariants? |
| [Features and models](features_and_models.md) | Which signals are built, and how are walk-forward scores produced? |
| [Research protocol](research_protocol.md) | What makes an experiment defensible? |
| [Backtesting](backtesting.md) | How are decisions, fills, costs, and constraints modeled? |
| [Broker integration](broker.md) | How is the separately invoked kabuS cash-equity adapter configured and operated? |
| [Compliance](compliance.md) | What are the licensing, privacy, security, and trading boundaries? |
| [Troubleshooting](troubleshooting.md) | What does a common validation or runtime error mean? |
| [Development](development.md) | How should contributors test and extend the implementation? |

## Four rules to remember

1. A successful sample run proves the pipeline works; it does not prove a strategy works.
2. Every feature input must have been available by its decision timestamp.
3. Raw source data is immutable; derived artifacts belong to a unique run.
4. Paper output is simulated or intent-only. Only the separate, explicitly invoked broker commands
   can route an order.

Return to the [project README](../README.md).
