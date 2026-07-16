from __future__ import annotations

from pathlib import Path


def test_agent_sidecar_source_has_no_broker_paper_or_execution_imports() -> None:
    root = Path("src/nlp_trader/research_agents")
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py")))

    for forbidden in (
        "from nlp_trader import pipeline",
        "from nlp_trader.pipeline",
        "from nlp_trader.paper",
        "from nlp_trader.portfolio",
        "from nlp_trader.backtest",
        "from nlp_trader.broker",
    ):
        assert forbidden not in source
