from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nlp_trader.nlp.local_generation import (
    LlamaCppGenerationSession,
    LocalGenerationConfig,
)
from nlp_trader.research_agents.config import (
    load_research_agent_config,
    require_enabled,
)
from nlp_trader.research_agents.prompts import action_schema
from nlp_trader.research_agents.registry import ResearchRegistryLedger
from nlp_trader.research_agents.runner import (
    replay_agent_run,
    run_research_agent,
    scrub_agent_environment,
    verify_stored_run,
)
from nlp_trader.research_agents.views import load_development_view_bundle

app = typer.Typer(
    name="nlp-trader-agent",
    help="Bounded local analyst sidecar over sealed development views.",
    add_completion=False,
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", dir_okay=False, readable=True, help="Agent-only YAML config."),
]


@app.command("propose")
def propose_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    attempt_id: Annotated[str, typer.Option("--attempt-id", help="Reserved attempt SHA-256.")],
    bundle_root: Annotated[
        Path,
        typer.Option("--bundle-root", file_okay=False, readable=True, help="Sealed bundle root."),
    ],
    config: ConfigOption = Path("configs/research_agent.disabled.yaml"),
) -> None:
    """Run one bounded local proposal attempt after all deterministic checks pass."""

    settings = load_research_agent_config(config)
    require_enabled(settings)
    assert settings.model_path is not None
    assert settings.model_expected_sha256 is not None
    ledger = ResearchRegistryLedger(settings.artifact_root)
    study = ledger.study_definition(study_id)
    bundle = load_development_view_bundle(bundle_root.expanduser().resolve())
    scrub_agent_environment()
    session = LlamaCppGenerationSession(
        LocalGenerationConfig(
            model_path=settings.model_path,
            expected_model_sha256=settings.model_expected_sha256,
            context_tokens=settings.context_tokens,
            prompt_batch_tokens=settings.prompt_batch_tokens,
            max_input_tokens=settings.max_input_tokens,
            max_new_tokens=settings.max_output_tokens,
            gpu_layers=settings.gpu_layers,
            flash_attention=settings.flash_attention,
            use_mmap=settings.use_mmap,
            seed=settings.seed,
            decoding=settings.decoding,
        ),
        action_schema(),
    )
    result = run_research_agent(
        config=settings,
        ledger=ledger,
        study=study,
        bundle=bundle,
        attempt_id=attempt_id,
        generator=session.generate,
        device_path=session.diagnostics.device,
        effective_gpu_layers=session.diagnostics.effective_gpu_layers,
    )
    typer.echo(f"agent_run_id: {result.initial.agent_run_id}")
    typer.echo(f"run_dir: {result.run_dir}")
    typer.echo(f"outcome: {result.final.outcome}")


@app.command("verify")
def verify_command(
    study_id: Annotated[str, typer.Option("--study-id", help="Registered study SHA-256.")],
    run_dir: Annotated[
        Path, typer.Option("--run-dir", file_okay=False, readable=True, help="Stored agent run.")
    ],
    bundle_root: Annotated[
        Path,
        typer.Option("--bundle-root", file_okay=False, readable=True, help="Sealed bundle root."),
    ],
    config: ConfigOption = Path("configs/research_agent.disabled.yaml"),
) -> None:
    """Recompute deterministic terminal verification without loading a model."""

    settings = load_research_agent_config(config)
    ledger = ResearchRegistryLedger(settings.artifact_root)
    verification = verify_stored_run(
        run_dir,
        ledger=ledger,
        study=ledger.study_definition(study_id),
        bundle=load_development_view_bundle(bundle_root.expanduser().resolve()),
    )
    typer.echo(f"verification_id: {verification.verification_id}")
    typer.echo(f"passed: {str(verification.passed).lower()}")


@app.command("replay")
def replay_command(
    run_dir: Annotated[
        Path, typer.Option("--run-dir", file_okay=False, readable=True, help="Stored agent run.")
    ],
) -> None:
    """Validate the stored generation and round chain without a model call."""

    rounds = replay_agent_run(run_dir)
    typer.echo(f"rounds: {len(rounds)}")
    typer.echo(f"final_round_hash: {rounds[-1].round_id}")


def main() -> None:
    app()
