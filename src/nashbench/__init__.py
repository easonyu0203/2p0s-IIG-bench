"""Benchmark games and evaluation for Nash equilibrium solvers."""

import importlib.metadata

from nashbench.core import Game
from nashbench.core import State
from nashbench.core import TimeStep
from nashbench.core import auto_reset
from nashbench.exploitability import Evaluation
from nashbench.exploitability import evaluate
from nashbench.exploitability import exploitability
from nashbench.games import REGISTRY
from nashbench.games import make
from nashbench.policy import Mixture
from nashbench.policy import Policy
from nashbench.policy import uniform_random

__version__ = importlib.metadata.version("nashbench")

__all__ = [
    "REGISTRY",
    "Evaluation",
    "Game",
    "Mixture",
    "Policy",
    "State",
    "TimeStep",
    "auto_reset",
    "evaluate",
    "exploitability",
    "make",
    "uniform_random",
]
