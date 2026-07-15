from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from nlp_trader import pipeline
from nlp_trader.broker.cli import broker_app
from nlp_trader.config import ResearchConfig, RuntimeConfig, TransformerConfig, load_config
from nlp_trader.data.synthetic import generate_synthetic_fixture
from nlp_trader.logging import configure_logging

DEFAULT_CONFIG = Path("configs/sample.yaml")

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        dir_okay=False,
        readable=True,
        help="Typed local research configuration.",
    ),
]
LimitOption = Annotated[
    int | None,
    typer.Option(
        "--limit",
        min=1,
        help="Limit complete decision timestamps while preserving history and cross-sections.",
    ),
]
StartDateOption = Annotated[
    str | None,
    typer.Option("--start-date", help="Inclusive ISO date or timezone-aware timestamp."),
]
EndDateOption = Annotated[
    str | None,
    typer.Option("--end-date", help="Inclusive ISO date or timezone-aware timestamp."),
]
SymbolsOption = Annotated[
    list[str] | None,
    typer.Option(
        "--symbol",
        "--symbols",
        help="Restrict to a symbol; repeat for multiple symbols.",
    ),
]

PipelineCommand = Callable[[ResearchConfig], dict[str, Any]]

app = typer.Typer(
    name="nlp-trader",
    help="Local-first research pipeline with separately gated broker operations.",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(broker_app, name="broker")


@app.callback()
def _application(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Enable debug logging for this local run."),
    ] = False,
) -> None:
    configure_logging(verbose)


def _config_with_runtime_overrides(
    config_path: Path,
    *,
    limit: int | None,
    start_date: str | None,
    end_date: str | None,
    symbols: list[str] | None,
) -> ResearchConfig:
    config = load_config(config_path)
    updates: dict[str, Any] = {}
    if limit is not None:
        updates["limit"] = limit
    if start_date is not None:
        updates["start_date"] = start_date
    if end_date is not None:
        updates["end_date"] = end_date
    if symbols:
        updates["symbols"] = tuple(symbols)
    if not updates:
        return config
    runtime_payload = config.runtime.model_dump(mode="python")
    runtime_payload.update(updates)
    runtime = RuntimeConfig.model_validate(runtime_payload)
    config_payload = config.model_dump(mode="python")
    config_payload["runtime"] = runtime
    return ResearchConfig.model_validate(config_payload)


def _print_outputs(outputs: dict[str, Any]) -> None:
    for key, value in outputs.items():
        typer.echo(f"{key}: {value}")


def _run_pipeline_command(
    command: PipelineCommand,
    config_path: Path,
    *,
    limit: int | None,
    start_date: str | None,
    end_date: str | None,
    symbols: list[str] | None,
) -> None:
    config = _config_with_runtime_overrides(
        config_path,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
    )
    _print_outputs(command(config))


@app.command("validate-config")
def validate_config_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Validate typed configuration and required local input paths."""

    loaded = _config_with_runtime_overrides(
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )
    result = pipeline.validate(loaded)
    if not result["ok"]:
        for error in result["errors"]:
            typer.echo(str(error), err=True)
        raise typer.Exit(code=1)
    _print_outputs({"ok": True})


@app.command("ingest-market")
def ingest_market_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Ingest local market fixtures into an immutable research run."""

    _run_pipeline_command(
        pipeline.ingest_market,
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )


@app.command("ingest-text")
def ingest_text_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Ingest permitted local natural-language fixtures."""

    _run_pipeline_command(
        pipeline.ingest_text,
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )


@app.command("build-features")
def build_features_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Build point-in-time market and text features."""

    _run_pipeline_command(
        pipeline.build_features,
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )


@app.command("build-labels")
def build_labels_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Build forward labels separately from features."""

    _run_pipeline_command(
        pipeline.build_labels,
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )


@app.command("train")
def train_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Train deterministic baseline model families."""

    _run_pipeline_command(
        pipeline.train,
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )


@app.command("predict")
def predict_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Generate out-of-sample baseline predictions."""

    _run_pipeline_command(
        pipeline.predict,
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )


@app.command("backtest")
def backtest_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Run the hypothetical, cost-aware research backtest."""

    _run_pipeline_command(
        pipeline.backtest,
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )


@app.command("paper")
def paper_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Write pending simulation-only intents without fills or live routing."""

    _run_pipeline_command(
        pipeline.paper,
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )


@app.command("report")
def report_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
) -> None:
    """Write an auditable hypothetical research report."""

    _run_pipeline_command(
        pipeline.report,
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )


@app.command("smoke")
def smoke_command(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: LimitOption = None,
    start_date: StartDateOption = None,
    end_date: EndDateOption = None,
    symbol: SymbolsOption = None,
    enable_transformer_sentiment: Annotated[
        bool,
        typer.Option(
            "--enable-transformer-sentiment",
            help="Use configured local-only transformer sentiment for this smoke run.",
        ),
    ] = False,
) -> None:
    """Run the smallest end-to-end local research pipeline."""

    loaded = _config_with_runtime_overrides(
        config,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        symbols=symbol,
    )
    if enable_transformer_sentiment:
        transformer = TransformerConfig.model_validate(
            {**loaded.transformer.model_dump(), "enabled": True}
        )
        loaded = loaded.model_copy(update={"transformer": transformer})
    _print_outputs(pipeline.smoke(loaded))


@app.command("generate-synthetic")
def generate_synthetic_command(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", file_okay=False, help="New local fixture directory."),
    ],
    seed: Annotated[int, typer.Option("--seed", help="Deterministic random seed.")] = 7,
    session_count: Annotated[
        int,
        typer.Option("--session-count", min=3, help="Number of exchange sessions."),
    ] = 8,
    symbol: Annotated[
        list[str] | None,
        typer.Option("--symbol", help="Synthetic symbol; repeat for multiple symbols."),
    ] = None,
) -> None:
    """Generate deterministic, redistributable fixtures without network access."""

    paths = generate_synthetic_fixture(
        output_dir,
        seed=seed,
        session_count=session_count,
        symbols=tuple(symbol) if symbol else ("AAA", "BBB"),
    )
    _print_outputs(
        {
            "assets": paths.assets,
            "market_bars": paths.market_bars,
            "text_items": paths.text_items,
        }
    )


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the Typer application programmatically while preserving the legacy API."""

    result: object = app(
        args=list(argv) if argv is not None else None,
        prog_name="nlp-trader",
        standalone_mode=False,
    )
    return result if isinstance(result, int) else 0


def main() -> None:
    app(prog_name="nlp-trader")
