"""The benchmark games, by name."""

from collections.abc import Callable
import functools

from nashbench import core
from nashbench.games import dark_hex
from nashbench.games import goofspiel
from nashbench.games import kuhn_poker
from nashbench.games import leduc_poker
from nashbench.games import liars_dice
from nashbench.games import oshi_zumo
from nashbench.games import phantom_ttt

#: Constructor of each game, by name.
REGISTRY: dict[str, Callable[..., core.Game]] = {
    "kuhn_poker": kuhn_poker.KuhnPoker,
    "leduc_poker": leduc_poker.LeducPoker,
    "phantom_ttt": functools.partial(phantom_ttt.PhantomTTT, abrupt=False),
    "phantom_ttt_abrupt": functools.partial(
        phantom_ttt.PhantomTTT, abrupt=True
    ),
    "dark_hex3": functools.partial(dark_hex.DarkHex3, abrupt=False),
    "dark_hex3_abrupt": functools.partial(dark_hex.DarkHex3, abrupt=True),
    "goofspiel": goofspiel.Goofspiel,
    "liars_dice": liars_dice.LiarsDice,
    "oshi_zumo": oshi_zumo.OshiZumo,
}


def make(name: str, **kwargs) -> core.Game:
    """Returns the game named `name`, a key of `REGISTRY`.

    Args:
        name: Name of the game.
        **kwargs: Game parameters, such as `num_cards` of `"goofspiel"`.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown game {name!r}; choose from {list(REGISTRY)}."
        )
    return REGISTRY[name](**kwargs)
