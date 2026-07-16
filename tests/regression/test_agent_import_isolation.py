from __future__ import annotations

import subprocess
import sys


def test_agent_cli_help_imports_no_execution_or_optional_model_modules() -> None:
    denied = (
        "nlp_trader.pipeline",
        "nlp_trader.paper",
        "nlp_trader.portfolio",
        "nlp_trader.backtest",
        "nlp_trader.broker",
        "llama_cpp",
    )
    code = "\n".join(
        (
            "import sys",
            "from nlp_trader.research_agents.cli import app",
            "from typer.testing import CliRunner",
            "result = CliRunner().invoke(app, ['--help'])",
            "assert result.exit_code == 0, result.output",
            f"denied = {denied!r}",
            "loaded = [name for name in sys.modules if any(",
            "    name == item or name.startswith(item + '.') for item in denied",
            ")]",
            "assert not loaded, loaded",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
