# Benchmark results

This page reports game sizes, exact exploitability evaluation time, and
simulation throughput. Times are medians of 10 runs.

## Exact exploitability evaluation

The following table reports the number of information sets and terminal
histories for each game, followed by exact exploitability evaluation time
and peak GPU memory. Information sets include both seats. Evaluations use a
randomly initialized network with two hidden layers of 256 units. The first
evaluation builds the sequence form and compiles the computation. Later
evaluations reuse the same game object and its compiled computation, with
new network parameters each time.

| Game | Information sets | Terminal histories | First evaluation | Later evaluations | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| [`kuhn_poker`](games/kuhn-poker.md) | 12 | 30 | TBD | TBD | TBD |
| [`leduc_poker`](games/leduc-poker.md) | 936 | 5,520 | TBD | TBD | TBD |
| [`goofspiel`](games/goofspiel.md) | 23,050,572 | 373,248,000 | TBD | TBD | TBD |
| [`liars_dice`](games/liars-dice.md) | 15,728,640 | 655,359,375 | TBD | TBD | TBD |
| [`oshi_zumo`](games/oshi-zumo.md) | 31,502,916 | 15,905,730 | TBD | TBD | TBD |
| [`phantom_ttt`](games/phantom-tic-tac-toe.md) | 5,990,669 | 9,829,101,024 | TBD | TBD | TBD |
| [`phantom_ttt_abrupt`](games/phantom-tic-tac-toe.md) | 23,310,269 | 13,578,403,440 | TBD | TBD | TBD |
| [`dark_hex3`](games/dark-hex.md) | 6,072,917 | 9,469,697,760 | TBD | TBD | TBD |
| [`dark_hex3_abrupt`](games/dark-hex.md) | 27,325,277 | 14,663,760,672 | TBD | TBD | TBD |
| [`universal_poker`](games/universal-poker.md) | 9,112,032 | 108,134,928 | TBD | TBD | TBD |
| [`battleship`](games/battleship.md) | 5,592,302 | 5,706,547,200 | TBD | TBD | TBD |
| [`blotto`](games/blotto.md) | 2 | 1,002,001 | TBD | TBD | TBD |

For how exact exploitability is computed, see [Evaluation](evaluation.md).

## Simulation throughput

The following table reports simulation throughput in **million environment
steps per second (M steps/s)**. Each column is the number of independent
environments stepped together. Values exclude JAX compilation and include
sampling uniformly random legal actions, stepping with
`jax.jit(jax.vmap(nashbench.auto_reset(game)))`, and computing the
observation of the player to act. In Goofspiel, Oshi-Zumo, and Blotto, a
turn takes two steps, one per seat.

| Game | 64 environments (M steps/s) | 1,024 environments (M steps/s) | 16,384 environments (M steps/s) |
| --- | ---: | ---: | ---: |
| `kuhn_poker` | TBD | TBD | TBD |
| `leduc_poker` | TBD | TBD | TBD |
| `goofspiel` | TBD | TBD | TBD |
| `liars_dice` | TBD | TBD | TBD |
| `oshi_zumo` | TBD | TBD | TBD |
| `phantom_ttt` | TBD | TBD | TBD |
| `phantom_ttt_abrupt` | TBD | TBD | TBD |
| `dark_hex3` | TBD | TBD | TBD |
| `dark_hex3_abrupt` | TBD | TBD | TBD |
| `universal_poker` | TBD | TBD | TBD |
| `battleship` | TBD | TBD | TBD |
| `blotto` | TBD | TBD | TBD |

## Measurement conditions

[`benchmarks/measure.py`](../benchmarks/measure.py) produces the table values
with JAX 0.11 on an NVIDIA RTX A6000 GPU. Each measurement runs in a new
process. See [its README](../benchmarks/README.md).
