# Benchmark results

This page reports game sizes, exact exploitability evaluation time, and
simulation throughput. All values are the mean plus or minus the standard
deviation of 10 runs on one NVIDIA RTX A6000 GPU.

## Exact exploitability evaluation

The following table reports the number of information sets and terminal
histories for each game, followed by exact exploitability evaluation time.
Information sets include both seats. The first evaluation builds the sequence
form and compiles the computation. Later evaluations reuse the same game
object and its compiled computation.

| Game | Information sets | Terminal histories | First evaluation (mean ± SD) | Later evaluations (mean ± SD) |
| --- | ---: | ---: | ---: | ---: |
| [`kuhn_poker`](games/kuhn-poker.md) | 12 | 30 | 1.1 ± 0.077 s | 1.4 ± 0.15 ms |
| [`leduc_poker`](games/leduc-poker.md) | 936 | 5,520 | 1.5 ± 0.072 s | 1.4 ± 0.13 ms |
| [`goofspiel`](games/goofspiel.md) | 23,050,572 | 373,248,000 | 2.2 ± 0.072 s | 86 ± 1.4 ms |
| [`liars_dice`](games/liars-dice.md) | 15,728,640 | 655,359,375 | 2.2 ± 0.47 s | 57 ± 5.9 ms |
| [`oshi_zumo`](games/oshi-zumo.md) | 31,502,916 | 15,905,730 | 2.5 ± 0.18 s | 0.18 ± 0.0002 s |
| [`phantom_ttt`](games/phantom-tic-tac-toe.md) | 5,990,669 | 9,829,101,024 | 13 ± 0.2 s | 5.5 ± 0.006 s |
| [`phantom_ttt_abrupt`](games/phantom-tic-tac-toe.md) | 23,310,269 | 13,578,403,440 | 18 ± 0.18 s | 8.1 ± 0.0041 s |
| [`dark_hex3`](games/dark-hex.md) | 6,072,917 | 9,469,697,760 | 13 ± 0.072 s | 5.4 ± 0.0036 s |
| [`dark_hex3_abrupt`](games/dark-hex.md) | 27,325,277 | 14,663,760,672 | 19 ± 0.1 s | 9.1 ± 0.0035 s |

For how exact exploitability is computed, see [Evaluation](evaluation.md).

## Simulation throughput

The following table reports simulation throughput in **million environment
steps per second (M steps/s)**. Each column is the number of independent
environments stepped together. Values exclude JAX compilation and include
sampling uniformly random legal actions, stepping with
`jax.jit(jax.vmap(nashbench.auto_reset(game)))`, and computing the
observation of the player to act. In Goofspiel and Oshi-Zumo, a turn takes
two steps, one per seat.

| Game | 64 environments (M steps/s) | 1,024 environments (M steps/s) | 16,384 environments (M steps/s) |
| --- | ---: | ---: | ---: |
| `kuhn_poker` | 3.9 ± 0.0044 | 60 ± 0.081 | 590 ± 0.7 |
| `leduc_poker` | 2.6 ± 0.0015 | 39 ± 0.011 | 470 ± 0.63 |
| `goofspiel` | 2.8 ± 0.0015 | 37 ± 0.022 | 300 ± 0.96 |
| `liars_dice` | 2.9 ± 0.0016 | 43 ± 0.03 | 410 ± 1.4 |
| `oshi_zumo` | 3.5 ± 0.0027 | 43 ± 0.021 | 210 ± 0.4 |
| `phantom_ttt` | 3.2 ± 0.0025 | 47 ± 0.047 | 330 ± 0.94 |
| `phantom_ttt_abrupt` | 3.3 ± 0.0015 | 46 ± 0.02 | 330 ± 1.3 |
| `dark_hex3` | 3.1 ± 0.0012 | 46 ± 0.027 | 290 ± 1.2 |
| `dark_hex3_abrupt` | 3.3 ± 0.001 | 46 ± 0.013 | 300 ± 1.1 |

## Measurement conditions

[`benchmarks/stats.py`](../benchmarks/stats.py) produces the table values
with JAX 0.11 on an NVIDIA RTX A6000 GPU. It compiles each simulation
workload before timing it. For exact evaluation, a first-call run starts with
empty JAX caches and a new game object; later-call runs reuse a game object
after one warmup evaluation. Exact evaluation of every game peaks below
8 GiB of GPU memory, including a two-layer policy network, so it also runs on
GPUs with 24 GB.
