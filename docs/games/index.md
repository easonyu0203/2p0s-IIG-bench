# Games

Each game replicates an OpenSpiel game. Pass its name to `nashbench.make`.

| Name | OpenSpiel game | Actions | Observation size |
| --- | --- | --- | --- |
| `kuhn_poker` | `kuhn_poker` | 2 | 11 |
| `leduc_poker` | `leduc_poker` | 3 | 30 |
| `phantom_ttt` | `phantom_ttt` | 9 | 110 |
| `phantom_ttt_abrupt` | `phantom_ttt(gameversion=abrupt)` | 9 | 110 |
| `dark_hex3` | `dark_hex` | 9 | 164 |
| `dark_hex3_abrupt` | `dark_hex(gameversion=adh)` | 9 | 164 |
| `goofspiel` | `goofspiel(players=2,num_cards=6,imp_info=true)` | 6 | 136 |

To load the OpenSpiel equivalent, pass the OpenSpiel game string to
`pyspiel.load_game`.

## Size

The size of a game tree determines how long exact evaluation takes; see
[Evaluation](../evaluation.md#exploitability). Information sets are counted
for both seats.

| Name | Information sets | Terminal histories |
| --- | --- | --- |
| `kuhn_poker` | 12 | 30 |
| `leduc_poker` | 936 | 5,520 |
| `phantom_ttt` | 5,990,669 | 9,829,101,024 |
| `phantom_ttt_abrupt` | 23,310,269 | 13,578,403,440 |
| `dark_hex3` | 6,072,917 | 9,469,697,760 |
| `dark_hex3_abrupt` | 27,325,277 | 14,663,760,672 |
| `goofspiel` | 23,050,572 | 373,248,000 |

```{toctree}
:maxdepth: 1

kuhn_poker
leduc_poker
phantom_ttt
dark_hex
goofspiel
```
