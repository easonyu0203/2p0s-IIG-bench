# Games

Each game replicates an OpenSpiel game. Pass its name to `nashbench.make`.
To load the OpenSpiel equivalent, pass the OpenSpiel game string to
`pyspiel.load_game`.

| Name | OpenSpiel game | Actions | Observation size |
| --- | --- | --- | --- |
| [`kuhn_poker`](kuhn_poker.md) | `kuhn_poker` | 2 | 11 |
| [`leduc_poker`](leduc_poker.md) | `leduc_poker` | 3 | 30 |
| [`goofspiel`](goofspiel.md) | `goofspiel(players=2,num_cards=6,imp_info=true)` | 6 | 136 |
| [`phantom_ttt`](phantom_ttt.md) | `phantom_ttt` | 9 | 110 |
| [`phantom_ttt_abrupt`](phantom_ttt.md) | `phantom_ttt(gameversion=abrupt)` | 9 | 110 |
| [`dark_hex3`](dark_hex.md) | `dark_hex` | 9 | 164 |
| [`dark_hex3_abrupt`](dark_hex.md) | `dark_hex(gameversion=adh)` | 9 | 164 |

For the size of each game tree and how long its exact evaluation takes, see
the [README](../../README.md).

## Speed

The table lists how many million steps per second each game runs, for a batch
of games stepped together. Each step runs
`jax.jit(jax.vmap(nashbench.auto_reset(game)))` with uniformly random legal
actions, inside `jax.lax.scan`, and computes both players' observations.
Measured with JAX 0.11 on one NVIDIA RTX A6000, the GPU for every
measurement in these docs, by
[`benchmarks/stats.py`](../../benchmarks/stats.py).

| Name | 1,024 games | 16,384 games | 262,144 games |
| --- | --- | --- | --- |
| `kuhn_poker` | 64 | 710 | 1200 |
| `leduc_poker` | 39 | 420 | 630 |
| `goofspiel` | 43 | 220 | 240 |
| `phantom_ttt` | 47 | 260 | 250 |
| `phantom_ttt_abrupt` | 47 | 270 | 280 |
| `dark_hex3` | 48 | 230 | 230 |
| `dark_hex3_abrupt` | 47 | 230 | 240 |

A single game takes 15 to 24 microseconds per step, mostly to launch GPU
kernels. In large batches, writing observations to memory bounds the speed,
so games with larger observations run fewer steps per second.
