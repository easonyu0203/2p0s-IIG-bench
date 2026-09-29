# Oshi-Zumo

`nashbench.make("oshi_zumo")` replicates OpenSpiel's
[`oshi_zumo`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/oshi_zumo/oshi_zumo.h)
with 13 coins and a minimum bid of 1: `oshi_zumo(coins=13,min_bid=1)`. For 1
to 13 coins, pass `coins`, for example
`nashbench.make("oshi_zumo", coins=5)`. To change the field, pass `size`.

A wrestler stands in the middle of `2 * size + 3` positions, numbered from 0.
Each seat starts with `coins` coins. Each turn, both seats bid at the same
time and pay their bids. The higher bid pushes the wrestler one position:
seat 0 pushes it up, and seat 1 down. Equal bids don't move it. Seat 0 wins
when the wrestler reaches the last position, and seat 1 when it reaches
position 0. If both seats run out of coins first, the seat that pushed the
wrestler past the middle wins. Seats see both bids after each turn, so the
only hidden information is the opponent's current bid.

Both seats act at every step: `current_player` is `nashbench.BOTH`.

> [!NOTE]
> OpenSpiel's default Oshi-Zumo allows bids of 0, so a game can last up to
> its horizon of 1,000 turns, and its tree is too large for exact evaluation.
> With a minimum bid of 1, a game lasts at most `coins` turns.

## Actions

Action `i` bids `i` coins. A seat bids from 1 coin to all its coins, or 0
once it has none.

## Observation

`2 + 2(n + 1) + 2s + 3 + 2n(n + 1)` dimensions, where `n` is `coins` and `s`
is `size`. For the default, 403 dimensions:

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2–15 | Seat 0's coins, one-hot over 0 to 13. |
| 16–29 | Seat 1's coins, one-hot over 0 to 13. |
| 30–38 | Wrestler's position, one-hot over 0 to 8. |
| 39–220 | Seat 0's bid in each of the 13 turns, one-hot over 0 to 13. Future turns are zeros. |
| 221–402 | Seat 1's bids, encoded like seat 0's. |

> [!NOTE]
> OpenSpiel's `oshi_zumo` has no information-state tensor. Dimensions 2–38
> are its observation tensor, which doesn't show how the game got there.
> nashbench appends every bid, so that observations identify information
> states, like OpenSpiel's information-state strings.

## Rewards

1 for a win, −1 for a loss, and 0 for a draw.
