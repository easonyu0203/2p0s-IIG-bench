# Goofspiel

`nashbench.make("goofspiel")` replicates OpenSpiel's
[`goofspiel`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/goofspiel/goofspiel.h)
with imperfect information: `goofspiel(players=2,num_cards=6,imp_info=true)`.
For 3 to 6 cards, pass `num_cards`, for example
`nashbench.make("goofspiel", num_cards=4)`.

Each seat holds bid cards worth 1 to `k`, and a deck of point cards worth 1 to
`k` is shuffled. Each turn, the top point card is revealed and both seats bid
a card at the same time. The higher bid wins the point card; equal bids
discard it. Seats see who won each turn, but not the opponent's bids. After
`k` turns, the seat with more points wins. As in OpenSpiel, the last turn has
a single legal bid and plays itself, so each seat makes `k - 1` decisions.

Seat 0 bids first, and seat 1 bids without seeing seat 0's bid. See
[Simultaneous moves](../core-concepts.md#simultaneous-moves).

> [!NOTE]
> OpenSpiel's default Goofspiel, with `imp_info=false`, reveals both bids after
> each turn. Its information states don't record the order of a seat's own
> bids, so they lack perfect recall. nashbench implements the imperfect
> information variant, the one commonly used as a benchmark.

## Actions

Action `i` bids the card worth `i + 1`. A card is legal if the seat still
holds it.

## Observation

`2 + 2(m + 1) + 3k + 2k^2` dimensions, where `m = k(k + 1) / 2` is the most
points a seat can win. For 6 cards, 136 dimensions:

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2–23 | Own points, one-hot over 0 to 21. |
| 24–45 | Opponent's points, one-hot over 0 to 21. |
| 46–51 | Own cards not yet bid. |
| 52–63 | Winner of each turn, one-hot over (seat 0, seat 1). Ties and future turns are zeros. |
| 64–99 | Point card of each turn, one-hot. Unrevealed cards are zeros. |
| 100–135 | Own bid in each turn, one-hot. Future turns are zeros. |

## Rewards

1 for more points, −1 for fewer, and 0 for a tie.
