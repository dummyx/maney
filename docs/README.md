# Documentation

This page is the map for NLP Trader’s documentation. You do not need to read everything before
running the sample.

## Pick a reading path

### I want to run it

1. [Getting started](getting_started.md)
2. [Configuration reference](configuration.md)
3. [Input data guide](input_data.md)
4. [Workflows and commands](workflows.md)
5. [Outputs and artifacts](outputs.md)
6. [Troubleshooting](troubleshooting.md)

### I want to evaluate a strategy

1. [Concepts and glossary](glossary.md)
2. [Data contracts](data_contracts.md)
3. [Features and models](features_and_models.md)
4. [Research protocol](research_protocol.md)
5. [Backtesting assumptions](backtesting.md)
6. [Compliance and safety](compliance.md)

### I want to change the code

1. [Architecture](architecture.md)
2. [Development guide](development.md)
3. [Data contracts](data_contracts.md)
4. [Research protocol](research_protocol.md)
5. [`AGENTS.md`](../AGENTS.md) for repository guardrails

## Document index

| Document | What it answers |
|---|---|
| [Getting started](getting_started.md) | How do I install, run, and inspect the synthetic sample? |
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
| [Compliance](compliance.md) | What are the licensing, privacy, security, and trading boundaries? |
| [Troubleshooting](troubleshooting.md) | What does a common validation or runtime error mean? |
| [Development](development.md) | How should contributors test and extend the implementation? |

## Four rules to remember

1. A successful sample run proves the pipeline works; it does not prove a strategy works.
2. Every feature input must have been available by its decision timestamp.
3. Raw source data is immutable; derived artifacts belong to a unique run.
4. Paper output is simulated or intent-only. Nothing in this repository routes an order.

Return to the [project README](../README.md).
