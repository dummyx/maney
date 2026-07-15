"""Deterministic, cost-aware research backtesting components."""

from nlp_trader.backtest.costs import CostBreakdown, CostModel, cost_breakdown
from nlp_trader.backtest.engine import run_backtest
from nlp_trader.backtest.metrics import summarize_backtest

__all__ = ["CostBreakdown", "CostModel", "cost_breakdown", "run_backtest", "summarize_backtest"]
