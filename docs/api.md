# API

The top-level `nashbench` package exports the public API, for example
`nashbench.Game`. Each name is documented by its docstring, which you can read
in the source below or with `help(nashbench.Game)`.

| Module | Names |
| --- | --- |
| [`games`](../src/nashbench/games/__init__.py) | `make`, `REGISTRY` |
| [`core`](../src/nashbench/core.py) | `Game`, `State`, `TimeStep`, `BOTH`, `auto_reset` |
| [`policy`](../src/nashbench/policy.py) | `Policy`, `uniform_random` |
| [`exploitability`](../src/nashbench/exploitability.py) | `exploitability` |
| [`sequence_form`](../src/nashbench/sequence_form.py) | `SequenceForm`, `enumerate_tree` (not exported) |

To make `exploitability` fast on a large game, override `Game.sequence_form`;
see [Contributing](../CONTRIBUTING.md#add-a-game).
