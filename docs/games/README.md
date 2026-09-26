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
