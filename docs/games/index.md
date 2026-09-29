# Games

Each game replicates an OpenSpiel game. Pass its name to `nashbench.make`.
To load the OpenSpiel equivalent, pass the OpenSpiel game string to
`pyspiel.load_game`.

| Name | OpenSpiel game | Actions | Observation size |
| --- | --- | --- | --- |
| [`kuhn_poker`](kuhn-poker.md) | `kuhn_poker` | 2 | 11 |
| [`leduc_poker`](leduc-poker.md) | `leduc_poker` | 3 | 30 |
| [`goofspiel`](goofspiel.md) | `goofspiel(players=2,num_cards=6,imp_info=true)` | 6 | 136 |
| [`liars_dice`](liars-dice.md) | `liars_dice(numdice=2,dice_sides=5)` | 21 | 33 |
| [`oshi_zumo`](oshi-zumo.md) | `oshi_zumo(coins=13,min_bid=1)` | 14 | 403 |
| [`phantom_ttt`](phantom-tic-tac-toe.md) | `phantom_ttt` | 9 | 110 |
| [`phantom_ttt_abrupt`](phantom-tic-tac-toe.md) | `phantom_ttt(gameversion=abrupt)` | 9 | 110 |
| [`dark_hex3`](dark-hex.md) | `dark_hex` | 9 | 164 |
| [`dark_hex3_abrupt`](dark-hex.md) | `dark_hex(gameversion=adh)` | 9 | 164 |

For game-tree sizes, exact evaluation times, and simulation throughput, see
[Benchmark results](../benchmarks.md).
