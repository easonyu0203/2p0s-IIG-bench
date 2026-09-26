"""Benchmark games and evaluation for Nash equilibrium solvers."""

import importlib.metadata

from nashbench.core import BOTH
from nashbench.core import Game
from nashbench.core import State
from nashbench.core import TimeStep
from nashbench.core import auto_reset
from nashbench.exploitability import exploitability
from nashbench.games import REGISTRY
from nashbench.games import make
from nashbench.policy import Policy
from nashbench.policy import uniform_random

__version__ = importlib.metadata.version("nashbench")

__all__ = [
    "BOTH",
    "REGISTRY",
    "Game",
    "Policy",
    "State",
    "TimeStep",
    "auto_reset",
    "exploitability",
    "make",
    "uniform_random",
]
