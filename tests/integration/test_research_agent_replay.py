from __future__ import annotations

from nlp_trader.research_agents.runner import scrub_agent_environment


def test_environment_scrub_removes_secret_values_without_retaining_them() -> None:
    secret_value = "planted-secret-value-never-retain"
    environment = {
        "PATH": "/usr/bin",
        "OMP_NUM_THREADS": "4",
        "VENDOR_API_SECRET": secret_value,
        "PYTHON_API_SECRET": secret_value,
        "GGML_API_SECRET": secret_value,
        "LLAMA_CPP_TOKEN": secret_value,
        "OMP_API_KEY": secret_value,
        "VECLIB_SECRET": secret_value,
        "UNRELATED_INHERITED_VALUE": secret_value,
    }

    removed = scrub_agent_environment(environment)

    assert "VENDOR_API_SECRET" in removed
    assert "PYTHON_API_SECRET" in removed
    assert "GGML_API_SECRET" in removed
    assert "LLAMA_CPP_TOKEN" in removed
    assert "OMP_API_KEY" in removed
    assert "VECLIB_SECRET" in removed
    assert "UNRELATED_INHERITED_VALUE" in removed
    assert environment["OMP_NUM_THREADS"] == "4"
    assert secret_value not in repr(removed)
    assert secret_value not in repr(environment)
