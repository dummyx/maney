"""Import-isolated package for the governed local research-agent sidecar.

The package root stays inert and dependency-light. Model generation, sealed-view tools, proposal
compilation, approved development execution, and one-time holdout evaluation live in explicit
submodules and commands; importing :mod:`nlp_trader.research_agents` grants no trading authority and
performs no model load, network access, or artifact mutation.
"""
