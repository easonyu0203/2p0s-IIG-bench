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
| [`kuhn_poker`](games/kuhn-poker.md) | 12 | 30 | 1.3 ± 0.081 s | 1.4 ± 0.23 ms |
| [`leduc_poker`](games/leduc-poker.md) | 936 | 5,520 | 1.8 ± 0.081 s | 1.9 ± 0.17 ms |
| [`goofspiel`](games/goofspiel.md) | 23,050,572 | 373,248,000 | 2.7 ± 0.13 s | 87 ± 1.1 ms |
| [`liars_dice`](games/liars-dice.md) | 15,728,640 | 655,359,375 | 2.2 ± 0.45 s | 60 ± 2.9 ms |
| [`oshi_zumo`](games/oshi-zumo.md) | 31,502,916 | 15,905,730 | 2.5 ± 0.15 s | 0.18 ± 0.0002 s |
| [`phantom_ttt`](games/phantom-tic-tac-toe.md) | 5,990,669 | 9,829,101,024 | 14 ± 0.42 s | 5.5 ± 0.013 s |
| [`phantom_ttt_abrupt`](games/phantom-tic-tac-toe.md) | 23,310,269 | 13,578,403,440 | 19 ± 0.75 s | 8.2 ± 0.017 s |
| [`dark_hex3`](games/dark-hex.md) | 6,072,917 | 9,469,697,760 | 14 ± 0.36 s | 5.4 ± 0.018 s |
| [`dark_hex3_abrupt`](games/dark-hex.md) | 27,325,277 | 14,663,760,672 | 21 ± 0.43 s | 9.5 ± 0.024 s |

For how exact exploitability is computed, see [Evaluation](evaluation.md).

## Simulation throughput

The following table reports simulation throughput in **million environment
steps per second (M steps/s)**. Each column is the number of independent
environments stepped together. Values exclude JAX compilation and include
sampling uniformly random legal actions, stepping with
`jax.jit(jax.vmap(nashbench.auto_reset(game)))`, and computing both players'
observations.

| Game | 64 environments (M steps/s) | 1,024 environments (M steps/s) | 16,384 environments (M steps/s) |
| --- | ---: | ---: | ---: |
| `kuhn_poker` | 3.7 ± 0.027 | 54 ± 2.3 | 430 ± 14 |
| `leduc_poker` | 2.5 ± 0.017 | 37 ± 0.27 | 400 ± 10 |
| `goofspiel` | 3.2 ± 0.012 | 44 ± 0.25 | 210 ± 7.6 |
| `liars_dice` | 2.8 ± 0.0011 | 42 ± 0.04 | 340 ± 0.61 |
| `oshi_zumo` | 3.7 ± 0.0035 | 39 ± 0.032 | 130 ± 0.25 |
| `phantom_ttt` | 3 ± 0.0035 | 44 ± 0.035 | 240 ± 0.81 |
| `phantom_ttt_abrupt` | 3.1 ± 0.042 | 44 ± 0.014 | 240 ± 12 |
| `dark_hex3` | 3 ± 0.0025 | 45 ± 0.022 | 210 ± 0.24 |
| `dark_hex3_abrupt` | 3 ± 0.13 | 44 ± 0.48 | 210 ± 8.8 |

## Measurement conditions

[`benchmarks/stats.py`](../benchmarks/stats.py) produces the table values
with JAX 0.11 on an NVIDIA RTX A6000 GPU. It compiles each simulation
workload before timing it. For exact evaluation, a first-call run starts with
empty JAX caches and a new game object; later-call runs reuse a game object
after one warmup evaluation. Exact evaluation of every game peaks below
8 GiB of GPU memory, including a two-layer policy network, so it also runs on
GPUs with 24 GB.
