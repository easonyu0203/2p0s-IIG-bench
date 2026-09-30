"""The sequence form: a policy-independent representation of a game tree.

In the sequence form, a strategy of a seat is its *realization plan*: for
each of the seat's sequences (information set and action), the probability
that the seat's own actions lead to it. Expected returns are then linear in
each seat's plan, which makes exact evaluation fast. See von Stengel, "Efficient
computation of behavior strategies", 1996.
"""

from collections.abc import Callable
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class SequenceForm:
    """The structure of a two-player zero-sum game, independent of policies.

    Information sets of both seats are numbered together: seat 0's first,
    then seat 1's, each ordered by level, the number of actions the seat took
    before reaching it. Sequences are numbered so that a vector over them
    reshapes to `[num_infosets + 1, num_actions]`: row 0 holds the empty
    sequences of seats 0 and 1, and row `i + 1` holds the actions at
    information set `i`.

    Attributes:
        levels: `(start, end)` ranges of information sets, one per level and
            seat, by increasing level.
        parent: `[num_infosets]` int32 sequence that leads to each information
            set.
        legal_action_mask: `[num_infosets, num_actions]` bool.
        observe: Maps an array of information sets to their observations.
        gradient: Maps a realization plan of both seats to its gradient. For
            each sequence, the gradient is the return of its seat, summed over
            terminal histories where it is the seat's last sequence, and
            weighted by chance and by the opponent's plan.
    """

    levels: tuple[tuple[int, int], ...]
    parent: jax.Array
    legal_action_mask: jax.Array
    observe: Callable[[jax.Array], jax.Array]
    gradient: Callable[[jax.Array], jax.Array]

    @functools.cached_property
    def num_infosets(self) -> tuple[int, int]:
        """The number of information sets of seats 0 and 1."""
        # Seat 1's start with its first level, whose parent is its empty
        # sequence, 1.
        total = self.parent.size
        seat_0 = min(
            (s for s, _ in self.levels if self.parent[s] == 1), default=total
        )
        return seat_0, total - seat_0


def enumerate_tree(game) -> SequenceForm:
    """Returns the sequence form of `game` by enumerating its tree.

    This works for games whose tree fits in memory, such as poker games.
    Observations must identify information states.
    """
    observe = _batched(game.observe)
    legal_action_mask = _batched(game.legal_action_mask)
    apply_action = _batched(game.apply_action)
    returns = _batched(game.returns)

    # Information sets are numbered per seat in discovery order for now, and a
    # seat's sequence is `num_actions * infoset + action`, or -1 if empty.
    infosets = ({}, {})
    parents, observations, masks = ([], []), ([], []), ([], [])
    last_sequences, weights = [], []

    states, probs = game.initial_states()
    sequences = np.full((len(probs), 2), -1)
    probs = np.asarray(probs)
    while len(probs):
        done = np.asarray(states.done)
        if done.any():
            last_sequences.append(sequences[done])
            weights.append(
                probs[done] * np.asarray(returns(_take(states, done)))[:, 0]
            )
            states, sequences, probs = _take((states, sequences, probs), ~done)
            if not len(probs):
                break
        seats = np.asarray(states.current_seat)
        obs = np.asarray(observe(states, seats))
        mask = np.asarray(legal_action_mask(states, seats))
        ids = np.empty(len(seats), int)
        for i, seat in enumerate(seats):
            key = obs[i].tobytes()
            if key not in infosets[seat]:
                infosets[seat][key] = len(parents[seat])
                parents[seat].append(sequences[i, seat])
                observations[seat].append(obs[i])
                masks[seat].append(mask[i])
            ids[i] = infosets[seat][key]
            if parents[seat][ids[i]] != sequences[i, seat]:
                raise ValueError(
                    "Observations don't identify information states."
                )
        node, action = np.nonzero(mask)
        states = apply_action(_take(states, node), action)
        sequences = sequences[node]
        sequences[np.arange(len(node)), seats[node]] = (
            ids[node] * mask.shape[1] + action
        )
        probs = probs[node]

    return _from_tables(
        parents,
        observations,
        masks,
        np.concatenate(last_sequences),
        np.concatenate(weights),
    )


def _batched(fn):
    """Vectorizes and jits `fn`, padding batches to powers of two.

    Padding bounds how many batch sizes `fn` compiles for.
    """
    jitted = jax.jit(jax.vmap(fn))

    def call(*args):
        args = jax.tree.map(np.asarray, args)
        n = len(jax.tree.leaves(args)[0])
        pad = max(1 << (n - 1).bit_length(), 2**12) - n
        args = jax.tree.map(
            lambda x: np.concatenate([x, x[:1].repeat(pad, 0)]), args
        )
        return jax.tree.map(lambda x: np.asarray(x)[:n], jitted(*args))

    return call


def _take(tree, index):
    """Indexes every array of `tree`."""
    return jax.tree.map(lambda x: x[index], tree)


def _from_tables(parents, observations, masks, last_sequences, weights):
    """Numbers information sets by level and builds the sequence form."""
    num_actions = masks[0][0].size
    num_infosets = [len(p) for p in parents]
    # Level of each information set, and its global number after sorting.
    levels, new_ids = [], []
    for seat in range(2):
        level = np.zeros(num_infosets[seat], int)
        for i, parent in enumerate(parents[seat]):
            level[i] = 0 if parent < 0 else level[parent // num_actions] + 1
        order = np.argsort(level, kind="stable")
        new_id = np.empty_like(order)
        new_id[order] = np.arange(len(order)) + seat * num_infosets[0]
        levels.append(level[order])
        new_ids.append((order, new_id))

    def to_global(sequence, seat):
        """Maps per-seat sequences (-1 if empty) to global ones."""
        sequence = np.asarray(sequence)
        infoset = new_ids[seat][1][np.maximum(sequence, 0) // num_actions]
        return np.where(
            sequence < 0,
            seat,
            num_actions * (infoset + 1) + sequence % num_actions,
        )

    parent = np.concatenate(
        [to_global(np.asarray(parents[s])[new_ids[s][0]], s) for s in range(2)]
    )
    observation = np.concatenate(
        [np.asarray(observations[s])[new_ids[s][0]] for s in range(2)]
    )
    mask = np.concatenate(
        [np.asarray(masks[s])[new_ids[s][0]] for s in range(2)]
    )
    level = np.concatenate(levels)
    bounds = []
    for lvl in range(level.max() + 1):
        for seat in range(2):
            start = seat * num_infosets[0]
            ids = np.nonzero(level[start : start + num_infosets[seat]] == lvl)[
                0
            ]
            if ids.size:
                bounds.append((start + ids[0], start + ids[-1] + 1))

    seq0 = jnp.asarray(to_global(last_sequences[:, 0], 0))
    seq1 = jnp.asarray(to_global(last_sequences[:, 1], 1))
    weights = jnp.asarray(weights, jnp.float32)

    @jax.jit
    def gradient(plan):
        g = jnp.zeros_like(plan).at[seq0].add(weights * plan[seq1])
        return g.at[seq1].add(-weights * plan[seq0])

    observation = jnp.asarray(observation)
    return SequenceForm(
        levels=tuple((int(s), int(e)) for s, e in bounds),
        parent=jnp.asarray(parent, jnp.int32),
        legal_action_mask=jnp.asarray(mask),
        observe=functools.partial(_rows, observation),
        gradient=gradient,
    )


@jax.jit
def _rows(array, ids):
    """Returns rows `ids` of `array`."""
    return array[ids]
